# File: train/train.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import yaml
import io
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Prevent GUI backend issues on headless/server mode
import matplotlib.patches as mpatches  # add this to the top of your script
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
import hashlib
import subprocess
import datetime

from data.dataset import GoogleEmbedDataset
from data.transform import RandomAugment
from models.resunet_vit import ResNetUNetViT
from train.metrics import compute_miou, compute_f1
from losses.focal_tversky import CombinedFocalTverskyLoss
from split_data import generate_splits_and_weights
import argparse
import gc

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

# ========= Crash Debugging =========
print("[DEBUG] CUDA devices available:", torch.cuda.device_count())
print("[DEBUG] CUDA memory allocated (MB):", torch.cuda.memory_allocated() / 1024**2)
print("[DEBUG] CUDA memory reserved (MB):", torch.cuda.memory_reserved() / 1024**2)

subprocess.run(["nvidia-smi"])

print(torch.cuda.get_device_name(0))
print("CUDA available:", torch.cuda.is_available())

# ========= Load config =========
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

timestamp = os.path.basename(config["processed_dir"])

if "{now}" in config["output_dir"]:
    config["output_dir"] = config["output_dir"].replace("{now}", timestamp)

print("[DEBUG] Training start time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def get_split_hash(cfg):
    split_cfg = cfg["splitting"]
    hash_input = json.dumps({
        "train_ratio": split_cfg.get("train_ratio", 0.8),
        "val_ratio": split_cfg.get("val_ratio", 0.1),
        "test_ratio": split_cfg.get("test_ratio", 0.1),
        "background_threshold": split_cfg.get("background_threshold", 0.9),
        "seed": split_cfg.get("seed", 42),
        "num_classes": cfg["num_classes"]
    }, sort_keys=True)
    return hashlib.md5(hash_input.encode()).hexdigest()[:8]

# ========= Load splits =========
split_hash = get_split_hash(config)
config["splits_path"] = f"data/splits/splits_{timestamp}_{split_hash}.json"

if not os.path.exists(config["splits_path"]):
    generate_splits_and_weights(config)

with open(config["splits_path"], "r") as f:
    splits = json.load(f)

# ========= TensorBoard =========
if config.get("tensorboard", {}).get("restart", True):
    port = config.get("tensorboard", {}).get("port", 6006)
    restart_tensorboard(logdir=config["output_dir"], port=port)

# ========= TensorBoard writer path =========
os.makedirs(config["output_dir"], exist_ok=True)
run_name = os.path.basename(config["output_dir"])
writer = SummaryWriter(log_dir=config["output_dir"])

# ========= Dataset =========
train_transform = RandomAugment(p=0.5)
train_ds = GoogleEmbedDataset(splits["train"], transform=train_transform)
val_ds   = GoogleEmbedDataset(splits["val"])

print(f"[INFO] Loaded {len(train_ds)} training and {len(val_ds)} validation samples.")
if len(train_ds) == 0 or len(val_ds) == 0:
    raise RuntimeError("Train or validation dataset is empty.")

# ========= Class weights and sample weights =========
n_classes = config["num_classes"]
print("→ Calculating pixel-wise class distribution for weighting...")

weights_dir = os.path.join(config["output_dir"], "weights")
os.makedirs(weights_dir, exist_ok=True)

weights_path = os.path.join(weights_dir, f"weights_{timestamp}_{split_hash}.npz")
pixel_path   = os.path.join(weights_dir, f"weights_{timestamp}_{split_hash}_pixels.npy")
os.makedirs(weights_dir, exist_ok=True)

# === 1. Pixel counts ===
if os.path.exists(pixel_path):
    print(f"[INFO] Found pixel counts at: {pixel_path}")
    label_counts = np.load(pixel_path)
else:
    print("→ Computing pixel-wise label counts...")
    label_counts = np.zeros(n_classes, dtype=np.int64)
    for base in tqdm(splits["train"], desc="Counting pixels"):
        lbl_path = base + "_lbl.npy"
        label = np.load(lbl_path, mmap_mode="r")
        mask = (label != 255) & (label < n_classes)
        valid = label[mask]
        count = np.bincount(valid.flatten(), minlength=n_classes)
        label_counts[:len(count)] += count
    np.save(pixel_path, label_counts)
    print(f"[INFO] Saved pixel counts to: {pixel_path}")

# === 2. Class/sample weights ===
if os.path.exists(weights_path):
    print(f"[INFO] Found class/sample weights at: {weights_path}")
    data = np.load(weights_path)
    class_weights = data["class_weights"]
    sample_weights = data["sample_weights"]
else:
    print("→ Computing class/sample weights...")
    class_weights = 1.0 / (label_counts + 1e-6)
    class_weights *= (n_classes / class_weights.sum())

    sample_weights = []
    for base in tqdm(splits["train"], desc="Computing sample weights"):
        lbl_path = base + "_lbl.npy"
        label = np.load(lbl_path, mmap_mode="r").flatten()
        label = label[(label != 255) & (label < len(class_weights))]
        weight = np.mean(class_weights[label]) if len(label) > 0 else 0.0
        sample_weights.append(weight)

    np.savez(weights_path, class_weights=class_weights, sample_weights=sample_weights)
    print(f"[INFO] Saved class/sample weights to: {weights_path}")


assert len(sample_weights) == len(train_ds.file_list), "Mismatch between weights and dataset entries"
rare_class_boost = config.get("sampling", {}).get("rare_class_boost", 1.5)

tile_scores = np.array(sample_weights)
tile_scores = tile_scores ** rare_class_boost
tile_scores /= tile_scores.sum()

sampler = WeightedRandomSampler(tile_scores, num_samples=len(train_ds.file_list), replacement=True)

# ========= Dataloaders =========
train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler,
                          num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
val_loader = DataLoader(
    val_ds,
    batch_size=config["batch_size"],  # unified
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

# ========= Model =========
model = ResNetUNetViT(config).cuda()
print("Device:", next(model.parameters()).device)

optimizer = optim.AdamW(model.parameters(),
                        lr=config["training"]["lr"],
                        weight_decay=config["training"]["weight_decay"])

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
ce_loss   = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=255)

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

            loss = ft_loss(out, y) + ce_loss(out, y)

        scaler.scale(loss).backward()
        clip_val = config.get("gradient_clipping", 1.0)  # Default to 1.0 if not specified
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

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

    acc = correct / total
    miou = compute_miou(all_preds, all_labels, config["num_classes"])
    f1   = compute_f1(all_preds, all_labels, config["num_classes"])

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, mIoU={miou:.4f}, F1={f1:.4f}")
    log_file.write(f"{epoch+1},{avg_loss:.4f},{acc:.4f},{miou:.4f},{f1:.4f}\n")
    log_file.flush()

    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/val", acc, epoch)
    writer.add_scalar("mIoU/val", miou, epoch)
    writer.add_scalar("F1/val", f1, epoch)
    writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

    # ===== Visual Logging to TensorBoard =====
    if (epoch + 1) % save_every == 0 or epoch == 0:
        model.eval()
        try:
            num_samples = 3
            fig, axs = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))

            for i in range(num_samples):
                sample_x, sample_y = val_ds[i]
                sample_x_vis = sample_x.clone()
                sample_x = sample_x.unsqueeze(0).cuda()
                with torch.no_grad():
                    pred = model(sample_x).argmax(1).squeeze().cpu()

                # Show selected embedding bands A01, A16, A09 = [0, 15, 8]
                vis_bands = [0, 15, 8]
                vis_img = sample_x_vis[vis_bands]
                vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min())
                vis_img = vis_img.permute(1, 2, 0).numpy()  # HWC

                axs[i, 0].imshow(vis_img)
                axs[i, 0].set_title("Google Embeddings (A01, A16, A09)")

                axs[i, 1].imshow(sample_y, cmap="tab20")
                axs[i, 1].set_title("Label")

                axs[i, 2].imshow(pred, cmap="tab20")
                axs[i, 2].set_title("Prediction")

                for j in range(3):
                    axs[i, j].axis("off")

            # Add legend to the last prediction panel
            class_colors = plt.cm.tab20(np.linspace(0, 1, config["num_classes"]))
            label_names = {}  # Ensure label_names is defined or loaded earlier
            class_labels = label_names if label_names else {i: f"Class {i}" for i in range(config["num_classes"])}
            legend_patches = [mpatches.Patch(color=class_colors[i], label=class_labels.get(str(i), f"Class {i}"))
                              for i in range(config["num_classes"])]
            axs[-1, 2].legend(handles=legend_patches, loc='upper right', fontsize='small', frameon=False)

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img = Image.open(buf)
            img_tensor = to_tensor(img)
            writer.add_image("Validation/Pred_vs_GT_Grid", img_tensor, global_step=epoch)
            plt.close("all")
            for ax in axs.flat:
                ax.clear()
            del fig, axs, vis_img, sample_x, sample_y, sample_x_vis, pred, img_tensor, img
            gc.collect()
            torch.cuda.empty_cache()
            writer.flush()

        except Exception as e:
            print(f"[WARN] TensorBoard image logging failed: {e}")

    if epoch % save_every == 0 and is_master_process():
        fp16_weights = convert_to_fp16(model.state_dict())
        torch.save(fp16_weights, os.path.join(config["output_dir"], f"model_epoch{epoch + 1}_weights.pt"), _use_new_zipfile_serialization=False)

    if epoch == 0:
        torch.cuda.empty_cache()
        gc.collect()

    curr_metric = avg_loss if es_metric == "loss" else f1
    improved = curr_metric < best_metric if es_metric == "loss" else curr_metric > best_metric

    if improved:
        best_metric = curr_metric
        no_improve = 0
        fp16_weights = convert_to_fp16(model.state_dict())
        if is_master_process():
            torch.save(fp16_weights, os.path.join(config["output_dir"], f"model_ep{epoch + 1}_weights.pt"),
                       _use_new_zipfile_serialization=False)
            torch.save(fp16_weights, os.path.join(config["output_dir"], "best_model_weights.pt"),
                       _use_new_zipfile_serialization=False)
            with open(os.path.join(config["output_dir"], "best_epoch.txt"), "w") as f:
                f.write(f"{epoch + 1},{f1:.4f}\n")
        print(f"[INFO] New best @ epoch {epoch + 1}, F1={f1:.4f}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch+1}")
            break

    scheduler.step(curr_metric)

    gc.collect()
    torch.cuda.empty_cache()

log_file.close()
writer.close()
