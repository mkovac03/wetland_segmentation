# File: train/train.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import binary_dilation

from data.dataset import GoogleEmbedDataset
from models.resunet_vit import ResNetUNetViT
from train.metrics import compute_miou, compute_f1

import argparse
import yaml

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0, reduction='mean', ignore_index=255, boundary_weight=0.2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.boundary_weight = boundary_weight

    def forward(self, inputs, targets):
        # Compute raw loss
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index
        )  # [B, H, W]
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # [B, H, W]

        # Create valid mask
        valid_mask = (targets != self.ignore_index).float()

        # Compute boundary mask
        boundary_mask = self._compute_boundary_mask(targets)  # float32, [B, H, W], values in {0.0, 1.0}
        weight_map = torch.where(boundary_mask.bool(), self.boundary_weight, 1.0).to(inputs.device)

        # Apply weighting
        focal_loss = focal_loss * valid_mask * weight_map

        if self.reduction == 'mean':
            return focal_loss.sum() / (valid_mask * weight_map).sum().clamp(min=1e-8)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def _compute_boundary_mask(self, targets):
        # targets: [B, H, W], long
        b, h, w = targets.shape
        boundary = torch.zeros((b, h, w), dtype=torch.float32)

        for i in range(b):
            t = targets[i].detach().cpu().numpy()
            t_mask = (t != self.ignore_index)
            t_dilated = binary_dilation(t_mask, iterations=1)
            edges = t_dilated & ~t_mask
            boundary[i] = torch.from_numpy(edges.astype(np.float32))

        return boundary

# Model, optimizer, loss
model = ResNetUNetViT(n_classes=config["num_classes"], input_channels=config["input_channels"]).cuda()
optimizer = optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
criterion = FocalLoss(alpha=1.0, gamma=1.0, ignore_index=255, boundary_weight=0.2)

scaler = GradScaler(enabled=config["training"]["use_amp"])

# Logging
os.makedirs(config["output_dir"], exist_ok=True)
log_path = os.path.join(config["output_dir"], "training_log.txt")
log_file = open(log_path, "a")
writer = SummaryWriter(log_dir=config["output_dir"])

# Training loop
for epoch in range(config["training"]["epochs"]):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        with autocast(enabled=config["training"]["use_amp"]):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
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

    # Logging
    log_file.write(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}, mIoU: {miou:.4f}, F1: {f1:.4f}\n")
    log_file.flush()
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/val", acc, epoch)
    writer.add_scalar("mIoU/val", miou, epoch)
    writer.add_scalar("F1/val", f1, epoch)

    # Save checkpoint
    ckpt_path = os.path.join(config["output_dir"], f"model_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), ckpt_path)

log_file.close()
writer.close()
