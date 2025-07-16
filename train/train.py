# File: train/train.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.dataset import GoogleEmbedDataset
from models.resunet_vit import ResNetUNetViT
from train.metrics import compute_miou, compute_f1
from losses.focal_tversky import CombinedFocalTverskyLoss

import torchvision.utils as vutils
import matplotlib.pyplot as plt
import io
from PIL import Image

import argparse
import yaml

print(torch.cuda.get_device_name(0))
print("CUDA available:", torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Load splits
with open(config["splits_path"], "r") as f:
    splits = json.load(f)

train_ds = GoogleEmbedDataset(splits["train"])
val_ds = GoogleEmbedDataset(splits["val"])

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# Model, optimizer, loss
model = ResNetUNetViT(n_classes=config["num_classes"], input_channels=config["input_channels"]).cuda()
print("Device:", next(model.parameters()).device)
optimizer = optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])

criterion = CombinedFocalTverskyLoss(
    num_classes=config["num_classes"],
    focal_alpha=config.get("loss", {}).get("focal", {}).get("alpha", 1.0),
    focal_gamma=config.get("loss", {}).get("focal", {}).get("gamma", 1.0),
    boundary_weight=config.get("loss", {}).get("focal", {}).get("boundary_weight", 0.2),
    tversky_alpha=config.get("loss", {}).get("tversky", {}).get("alpha", 0.5),
    tversky_beta=config.get("loss", {}).get("tversky", {}).get("beta", 0.3),
    ignore_index=255
)

scaler = GradScaler(enabled=config["training"].get("use_amp", False))

# Scheduler and early stopping
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=config.get("scheduler", {}).get("factor", 0.5),
    patience=config.get("scheduler", {}).get("patience", 5),
    verbose=True
)
patience = config["training"].get("early_stopping_patience", 10)
epochs_no_improve = 0

# Logging
os.makedirs(config["output_dir"], exist_ok=True)
log_path = os.path.join(config["output_dir"], "training_log.txt")
log_file = open(log_path, "a")
writer = SummaryWriter(log_dir=config["output_dir"])

# Training loop
best_f1 = -1
for epoch in range(config["training"]["epochs"]):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        with autocast(enabled=config["training"].get("use_amp", False)):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get("gradient_clipping", 1.0))

        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

    # Validation
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
    f1 = compute_f1(all_preds, all_labels, config["num_classes"])

    log_file.write(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}, mIoU: {miou:.4f}, F1: {f1:.4f}\n")
    log_file.flush()
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/val", acc, epoch)
    writer.add_scalar("mIoU/val", miou, epoch)
    writer.add_scalar("F1/val", f1, epoch)

    scheduler.step(f1)
    current_lr = optimizer.param_groups[0]["lr"]
    writer.add_scalar("LearningRate", current_lr, epoch)


    def create_image_grid(gt_tensor, pred_tensor, epoch, max_samples=5, num_classes=20):
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
        image = Image.open(buf)
        return image


    # --- Log visualization every 5 epochs ---
    if epoch % 5 == 0:
        n_samples = min(5, len(val_loader))
        val_iter = iter(val_loader)
        imgs, lbls, preds = [], [], []
        with torch.no_grad():
            for _ in range(n_samples):
                try:
                    x, y = next(val_iter)
                    x, y = x.cuda(), y.cuda()
                    out = model(x)
                    pred = out.argmax(1)
                    lbls.append(y.cpu())
                    preds.append(pred.cpu())
                except StopIteration:
                    break
        if preds:
            lbls_tensor = torch.cat(lbls)
            preds_tensor = torch.cat(preds)
            img = create_image_grid(lbls_tensor, preds_tensor, epoch, max_samples=n_samples,
                                    num_classes=config["num_classes"])
            writer.add_image("Samples/GT_vs_Pred", vutils.to_tensor(img), epoch)

    # Save best model and check early stopping
    if f1 > best_f1:
        best_f1 = f1
        epochs_no_improve = 0
        ckpt_path = os.path.join(config["output_dir"], f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

log_file.close()
writer.close()
