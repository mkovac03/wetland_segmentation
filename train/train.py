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
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor

from data.dataset import GoogleEmbedDataset
from data.transform import RandomFlipRotate
from models.resunet_vit_configurable import ResNetUNetViT
from train.metrics import compute_miou, compute_f1
from losses.focal_tversky import CombinedFocalTverskyLoss

import argparse

torch.backends.cudnn.benchmark = True

print(torch.cuda.get_device_name(0))
print("CUDA available:", torch.cuda.is_available())

# ========= Load config =========
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# ========= Load splits =========
with open(config["splits_path"], "r") as f:
    splits = json.load(f)

# ========= Dataset =========
train_transform = RandomFlipRotate(p=0.5)
train_ds = GoogleEmbedDataset(splits["train"], transform=train_transform)
val_ds   = GoogleEmbedDataset(splits["val"])

print(f"[INFO] Loaded {len(train_ds)} training and {len(val_ds)} validation samples.")
if len(train_ds) == 0 or len(val_ds) == 0:
    raise RuntimeError("Train or validation dataset is empty.")

# ========= Class weights and sample weights =========
n_classes = config["num_classes"]
print("→ Calculating pixel-wise class distribution for weighting...")
label_counts = np.zeros(n_classes, dtype=np.int64)
for i in tqdm(range(len(train_ds)), desc="Scanning training labels"):
    _, label = train_ds[i]
    label = label.numpy().flatten()
    label = label[label != 255]
    label_counts += np.bincount(label, minlength=n_classes)

class_weights = 1.0 / (label_counts + 1e-6)
class_weights = class_weights * (n_classes / class_weights.sum())

sample_weights = []
for i in range(len(train_ds)):
    _, label = train_ds[i]
    label = label.numpy().flatten()
    label = label[label != 255]
    sample_weights.append(class_weights[label].mean() if len(label) > 0 else 0.0)

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# ========= Dataloaders =========
train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler,
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# ========= Model =========
model = ResNetUNetViT(config).cuda()
print("Device:", next(model.parameters()).device)

optimizer = optim.AdamW(model.parameters(),
                        lr=config["training"]["lr"],
                        weight_decay=config["training"]["weight_decay"])

# ========= Losses =========
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

scaler = GradScaler(enabled=config["training"]["use_amp"])

# ========= Scheduler & Early Stopping =========
es_metric = config["training"]["early_stopping_metric"].lower()
best_metric = float("inf") if es_metric == "loss" else -1
patience = config["training"]["early_stopping_patience"]
no_improve = 0

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min" if es_metric == "loss" else "max",
    factor=config.get("scheduler", {}).get("factor", 0.5),
    patience=config.get("scheduler", {}).get("patience", 5),
    verbose=True
)

# ========= Logging =========
os.makedirs(config["output_dir"], exist_ok=True)
log_file = open(os.path.join(config["output_dir"], "training_log.txt"), "a")
writer = SummaryWriter(log_dir=config["output_dir"])

# ========= Training Loop =========
for epoch in range(config["training"]["epochs"]):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        with autocast(enabled=config["training"]["use_amp"]):
            out = model(x)
            loss = ft_loss(out, y) + ce_loss(out, y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])
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

    # ========= Visualization (every 5 epochs) =========
    if epoch % 5 == 0:
        def create_image_grid(gt_tensor, pred_tensor, max_samples=5, num_classes=20):
            fig, axes = plt.subplots(max_samples, 2, figsize=(6, 2 * max_samples))
            cmap = plt.get_cmap("tab20", num_classes)
            for i in range(max_samples):
                gt_img = gt_tensor[i].squeeze().numpy()
                pred_img = pred_tensor[i].squeeze().numpy()
                axes[i, 0].imshow(gt_img, cmap=cmap, vmin=0, vmax=num_classes - 1)
                axes[i, 0].set_title("Ground Truth")
                axes[i, 0].axis("off")
                axes[i, 1].imshow(pred_img, cmap=cmap, vmin=0, vmax=num_classes - 1)
                axes[i, 1].set_title("Prediction")
                axes[i, 1].axis("off")
            fig.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)

        val_iter = iter(val_loader)
        lbls, preds = [], []
        with torch.no_grad():
            for _ in range(min(5, len(val_loader))):
                try:
                    x, y = next(val_iter)
                    x, y = x.cuda(), y.cuda()
                    pred = model(x).argmax(1)
                    lbls.append(y.cpu())
                    preds.append(pred.cpu())
                except StopIteration:
                    break
        if preds:
            img = create_image_grid(torch.cat(lbls), torch.cat(preds), num_classes=config["num_classes"])
            writer.add_image("Samples/GT_vs_Pred", to_tensor(img), epoch)

    # ========= Early Stopping =========
    curr_metric = avg_loss if es_metric == "loss" else f1
    improved = curr_metric < best_metric if es_metric == "loss" else curr_metric > best_metric

    if improved:
        best_metric = curr_metric
        no_improve = 0
        torch.save(model.state_dict(), os.path.join(config["output_dir"], f"model_ep{epoch+1}.pt"))
        torch.save(model.state_dict(), os.path.join(config["output_dir"], "best_model.pt"))
        with open(os.path.join(config["output_dir"], "best_epoch.txt"), "w") as f:
            f.write(f"{epoch+1},{f1:.4f}\n")
        print(f"[INFO] New best @ epoch {epoch+1}, F1={f1:.4f}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch+1}")
            break

    scheduler.step(curr_metric)

log_file.close()
writer.close()
