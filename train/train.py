# File: train/train.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ========== Imports ==========
import json
import yaml
import io
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
import hashlib
import subprocess
import datetime
import psutil
import pynvml

# ========== Project Modules ==========
from data.dataset import GoogleEmbedDataset
from data.transform import RandomAugment
from models.resunet_vit import ResNetUNetViT
from train.metrics import compute_miou, compute_f1
from losses.focal_tversky import CombinedFocalTverskyLoss
from split_data import generate_splits_and_weights
from utils.plotting import add_prediction_legend
import argparse
import gc

# ========== Utility Functions ==========
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def log_memory(epoch):
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    ram = psutil.virtual_memory()
    print(f"[MEMORY][Epoch {epoch}] GPU: {gpu_mem.used // 1024**2} MB / {gpu_mem.total // 1024**2} MB, RAM: {ram.percent}% used")

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise ValueError("safe_collate: all samples in batch failed.")
    return default_collate(batch)

def is_master_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

def convert_to_fp16(state_dict):
    return {k: v.half() if v.dtype == torch.float32 else v for k, v in state_dict.items()}

def restart_tensorboard(logdir, port=6006):
    try:
        subprocess.run(["pkill", "-f", "tensorboard"], check=False)
        subprocess.run(["pkill", "-f", "tensorboard_data_server"], check=False)
        subprocess.Popen([
            "tensorboard",
            f"--logdir={logdir}",
            f"--port={port}",
            "--host=127.0.0.1",
            "--reload_interval=5",
            "--load_fast=false"
        ])
        print(f"[INFO] TensorBoard restarted at http://localhost:{port}")
    except Exception as e:
        print(f"[WARN] Failed to restart TensorBoard: {e}")

# ========== Load Config ==========
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
parser.add_argument("--resume", default=None)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

timestamp = os.path.basename(config["processed_dir"])
if "{now}" in config["output_dir"]:
    config["output_dir"] = config["output_dir"].replace("{now}", timestamp)

print("[DEBUG] Training start time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ========== Compute Split Hash ==========
def get_split_hash(cfg):
    split_cfg = cfg["splitting"]
    hash_input = json.dumps({
        "train_ratio": split_cfg.get("train_ratio", 0.8),
        "val_ratio": split_cfg.get("val_ratio", 0.1),
        "test_ratio": split_cfg.get("test_ratio", 0.1),
        "seed": split_cfg.get("seed", 42),
        "num_classes": cfg["num_classes"]
    }, sort_keys=True)
    return hashlib.md5(hash_input.encode()).hexdigest()[:8]

# ========== Load Splits ==========
split_hash = get_split_hash(config)
config["splits_path"] = f"data/splits/splits_{timestamp}_{split_hash}.json"
if not os.path.exists(config["splits_path"]):
    generate_splits_and_weights(config)

with open(config["splits_path"], "r") as f:
    splits = json.load(f)

# ========== TensorBoard ==========
if config.get("tensorboard", {}).get("restart", True):
    port = config.get("tensorboard", {}).get("port", 6006)
    restart_tensorboard(logdir=config["output_dir"], port=port)

writer = SummaryWriter(log_dir=config["output_dir"])

# ========== Datasets ==========
n_classes = config["num_classes"]
train_transform = RandomAugment(p=0.5)
expected_channels = config["input_channels"]

train_ds = GoogleEmbedDataset(
    splits["train"],
    transform=train_transform,
    num_classes=config["num_classes"],
    expected_channels=expected_channels
)

val_ds = GoogleEmbedDataset(
    splits["val"],
    num_classes=config["num_classes"],
    expected_channels=expected_channels
)

print(f"[INFO] Loaded {len(train_ds)} training and {len(val_ds)} validation samples.")
if len(train_ds) == 0 or len(val_ds) == 0:
    raise RuntimeError("Train or validation dataset is empty.")

# ========== Load Weights ==========
weights_path = os.path.join(config["output_dir"], "weights", f"weights_{timestamp}_{split_hash}.npz")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"[ERROR] Weights file not found: {weights_path}")

data = np.load(weights_path)
class_weights = data["class_weights"]
sample_weights = data["sample_weights"]
del data

if len(sample_weights) != len(train_ds.file_list):
    raise RuntimeError("Mismatch between sample weights and dataset tiles.")

# ========== Sampler ==========
tile_scores = np.array(sample_weights)
tile_scores = tile_scores ** config.get("sampling", {}).get("rare_class_boost", 1.5)
tile_scores /= tile_scores.sum()

sampler = WeightedRandomSampler(tile_scores, num_samples=len(train_ds.file_list), replacement=True)

# ========== Dataloaders ==========
train_loader = DataLoader(
    train_ds,
    batch_size=config["batch_size"],
    sampler=sampler,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    collate_fn=safe_collate
)

val_loader = DataLoader(
    val_ds,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    collate_fn=safe_collate
)

# ========== Model and Optimizer ==========
model = ResNetUNetViT(config).cuda()
optimizer = optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])

# ========== Loss Functions ==========
ft_loss = CombinedFocalTverskyLoss(
    num_classes=config["num_classes"],
    focal_alpha=config["loss"]["focal"]["alpha"],
    focal_gamma=config["loss"]["focal"]["gamma"],
    boundary_weight=config["loss"]["focal"]["boundary_weight"],
    tversky_alpha=config["loss"]["tversky"]["alpha"],
    tversky_beta=config["loss"]["tversky"]["beta"],
    ignore_index=255
)

ce_weight = torch.tensor(class_weights, dtype=torch.float32).cuda()
ce_loss = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=255)

# ========== AMP Scaler ==========
scaler = GradScaler(enabled=config["training"].get("use_amp", True))

es_metric = config["training"]["early_stopping_metric"].lower()
best_metric = float("inf") if es_metric == "loss" else -1
patience = config["training"]["early_stopping_patience"]
no_improve = 0

scheduler_cfg = config.get("scheduler", {})
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min" if es_metric=="loss" else "max",
    factor=scheduler_cfg.get("factor", 0.5),
    patience=scheduler_cfg.get("patience", 5)
)

log_file = open(os.path.join(config["output_dir"], "training_log.txt"), "a")
save_every = config.get("save_every", 5)

for epoch in range(config["training"]["epochs"]):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}"):
        gc.collect()
        torch.cuda.empty_cache()

        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        with autocast(device_type='cuda', enabled=config["training"].get("use_amp", True)):
            out = model(x)
            valid_mask = (y != 255) & (y < n_classes)
            if valid_mask.sum() == 0:
                continue  # skip empty batch

            y_valid = y[valid_mask]
            out_valid = out.permute(0, 2, 3, 1)[valid_mask]  # match shape [N, C]
            loss = ft_loss(out, y) + ce_loss(out_valid, y_valid)

        scaler.scale(loss).backward()
        clip_val = config.get("gradient_clipping", 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        # --- Memory cleanup ---
        del x, y, out, loss, y_valid, out_valid, valid_mask
        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = total_loss / len(train_loader)

    # ——— Validation ———
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            pred = out.argmax(1)
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())
            correct += (pred == y).sum().item()
            total += y.numel()

            del x, y, out, pred
            torch.cuda.empty_cache()
            gc.collect()

    acc = correct / total
    miou = compute_miou(all_preds, all_labels, config["num_classes"])
    f1 = compute_f1(all_preds, all_labels, config["num_classes"])

    # Add these to release large lists
    del all_preds, all_labels
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, mIoU={miou:.4f}, F1={f1:.4f}")
    log_file.write(f"{epoch+1},{avg_loss:.4f},{acc:.4f},{miou:.4f},{f1:.4f}\n")
    log_file.flush()

    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/val", acc, epoch)
    writer.add_scalar("mIoU/val", miou, epoch)
    writer.add_scalar("F1/val", f1, epoch)
    writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

    # ===== Convert and Save Model Weights if Needed =====
    fp16_weights = None
    curr_metric = avg_loss if es_metric == "loss" else f1
    improved = curr_metric < best_metric if es_metric == "loss" else curr_metric > best_metric

    if epoch % save_every == 0 or improved:
        fp16_weights = convert_to_fp16(model.state_dict())

        if is_master_process():
            if epoch % save_every == 0:
                torch.save(fp16_weights, os.path.join(config["output_dir"], f"model_ep{epoch + 1}_weights.pt"),
                           _use_new_zipfile_serialization=False)

            if improved:
                torch.save(fp16_weights, os.path.join(config["output_dir"], "best_model_weights.pt"),
                           _use_new_zipfile_serialization=False)
                with open(os.path.join(config["output_dir"], "best_epoch.txt"), "w") as f:
                    f.write(f"{epoch + 1},{f1:.4f}\n")

    if improved:
        best_metric = curr_metric
        no_improve = 0
        print(f"[INFO] New best @ epoch {epoch + 1}, F1={f1:.4f}")
        del acc, miou, f1, avg_loss
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch + 1}")
            break

    del fp16_weights
    gc.collect()
    torch.cuda.empty_cache()

    # ===== Visual Logging to TensorBoard =====
    if (epoch + 1) % save_every == 0 or epoch == 0:
        log_memory(f"{epoch}-before-TB")
        model.eval()
        try:
            num_samples = 1
            fig, axs = plt.subplots(num_samples, 2, figsize=(8, 3 * num_samples))  # 2 columns only

            for i in range(num_samples):
                tile_path = val_ds.file_list[i]
                tile_id = os.path.basename(tile_path)

                sample_x, sample_y = val_ds[i]
                sample_x = sample_x.unsqueeze(0).cuda()
                with torch.no_grad():
                    pred = model(sample_x).argmax(1).squeeze().cpu()

                # Label distribution printout
                u, c = np.unique(sample_y.numpy(), return_counts=True)
                dist_str = ", ".join([f"{int(cls)}: {cnt}" for cls, cnt in zip(u, c)])
                print(f"[VISUAL LOGGING] {tile_id} → Label Distribution: {dist_str}")

                axs[i, 0].imshow(sample_y, cmap="tab20")
                axs[i, 0].set_title(f"{tile_id} - Label")

                axs[i, 1].imshow(pred, cmap="tab20")
                axs[i, 1].set_title(f"{tile_id} - Prediction")

                for j in range(2):
                    axs[i, j].axis("off")

            label_names = config.get("label_names", {})
            add_prediction_legend(axs[-1, 1], config["num_classes"], label_names)

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img = Image.open(buf)
            img_tensor = to_tensor(img)
            writer.add_image("Validation/Label_vs_Prediction", img_tensor, global_step=epoch)
            plt.close("all")
            writer.flush()

            del fig, axs, sample_x, sample_y, pred, img_tensor, img, buf
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[WARN] TensorBoard image logging failed: {e}")

log_file.close()
writer.close()
