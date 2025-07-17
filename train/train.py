# File: train/train.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import yaml
import io
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import GoogleEmbedDataset
from models.resunet_vit import ResNetUNetViT
from train.metrics import compute_miou, compute_f1
from losses.focal_tversky import CombinedFocalTverskyLoss
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt

import argparse

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

train_ds = GoogleEmbedDataset(splits["train"])
val_ds   = GoogleEmbedDataset(splits["val"])

print(f"[INFO] Loaded {len(train_ds)} training and {len(val_ds)} validation samples.")
if len(train_ds)==0 or len(val_ds)==0:
    raise RuntimeError("Empty train/val set—check your preprocessing & splits.")

# ========= Build class‐balanced sampler =========
n_classes = config["num_classes"]
# 1) collect label of each sample
all_labels    = [ int(train_ds[i][1].item()) for i in range(len(train_ds)) ]
# 2) count per‐class freq
class_counts  = np.bincount(all_labels, minlength=n_classes)
# 3) inverse‐freq weights
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights * (n_classes / class_weights.sum())  # normalize
# 4) per‐sample weights
sample_weights = [ class_weights[l] for l in all_labels ]
# 5) sampler + loader
sampler      = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(train_ds,
                          batch_size=config["batch_size"],
                          sampler=sampler,
                          num_workers=4,
                          pin_memory=True)

val_loader   = DataLoader(val_ds,
                          batch_size=1,
                          shuffle=False,
                          num_workers=2,
                          pin_memory=True)

# ========= Model =========
model = ResNetUNetViT(n_classes=config["num_classes"],
                      input_channels=config["input_channels"]).cuda()
print("Device:", next(model.parameters()).device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=config["training"]["lr"],
    weight_decay=config["training"]["weight_decay"]
)

# ========= Losses =========
# 1) your combined Focal‐Tversky
ft_loss = CombinedFocalTverskyLoss(
    num_classes    = config["num_classes"],
    focal_alpha    = config["loss"]["focal"]["alpha"],
    focal_gamma    = config["loss"]["focal"]["gamma"],
    boundary_weight= config["loss"]["focal"]["boundary_weight"],
    tversky_alpha  = config["loss"]["tversky"]["alpha"],
    tversky_beta   = config["loss"]["tversky"]["beta"],
    ignore_index   = 255
)
# 2) a weighted CE to further punish the large‐bg class
ce_weight = torch.tensor(class_weights, dtype=torch.float32).cuda()
ce_loss    = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=255)

scaler = GradScaler(enabled=config["training"]["use_amp"])

# ========= Early stopping / Scheduler =========
es_metric    = config["training"]["early_stopping_metric"].lower()
best_metric  = float("inf") if es_metric=="loss" else -1
patience     = config["training"]["early_stopping_patience"]
no_improve   = 0

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min" if es_metric=="loss" else "max",
    factor=config.get("scheduler",{}).get("factor",0.5),
    patience=config.get("scheduler",{}).get("patience",5),
    verbose=True
)

# ========= Logging setup =========
os.makedirs(config["output_dir"], exist_ok=True)
log_file = open(os.path.join(config["output_dir"],"training_log.txt"),"a")
writer   = SummaryWriter(log_dir=config["output_dir"])

# ========= Training loop =========
for epoch in range(config["training"]["epochs"]):
    model.train()
    running_loss = 0.0

    for x,y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        with autocast(enabled=config["training"]["use_amp"]):
            out1 = model(x)
            l1   = ft_loss(out1, y)
            l2   = ce_loss(out1, y)
            loss = l1 + l2

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["gradient_clipping"])
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # —— validation ——
    model.eval()
    correct=0; total=0
    all_p, all_t = [], []
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.cuda(), y.cuda()
            out = model(x)
            pred= out.argmax(1)
            all_p.append(pred.cpu()); all_t.append(y.cpu())
            correct+= (pred==y).sum().item()
            total  += y.numel()

    acc  = correct/total
    miou = compute_miou(all_p, all_t, config["num_classes"])
    f1   = compute_f1(all_p, all_t, config["num_classes"])

    print(f"Epoch {epoch+1}, L={avg_loss:.4f}, Acc={acc:.4f}, miou={miou:.4f}, F1={f1:.4f}")
    log_file.write(f"{epoch+1},{avg_loss:.4f},{acc:.4f},{miou:.4f},{f1:.4f}\n")
    log_file.flush()

    # TensorBoard scalars
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Acc/val",   acc,       epoch)
    writer.add_scalar("mIoU/val",  miou,      epoch)
    writer.add_scalar("F1/val",    f1,        epoch)
    writer.add_scalar("LR",        optimizer.param_groups[0]["lr"], epoch)

    # ( … your “every‐5‐epochs” plotting stays the same … )

    # Early‐Stopping & checkpoint
    curr = avg_loss if es_metric=="loss" else f1
    better = curr<best_metric if es_metric=="loss" else curr>best_metric

    if better:
        best_metric=curr; no_improve=0
        # save both epoch and “best_model.pt”
        ckpt = os.path.join(config["output_dir"], f"model_ep{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt)
        torch.save(model.state_dict(), os.path.join(config["output_dir"],"best_model.pt"))
        with open(os.path.join(config["output_dir"],"best_epoch.txt"),"w") as f:
            f.write(f"{epoch+1},{f1:.4f}\n")
        print(f"[INFO] New best @ epoch {epoch+1}, F1={f1:.4f}")
    else:
        no_improve+=1
        if no_improve>=patience:
            print(f"[INFO] Early‐stopping @ epoch {epoch+1}")
            break

    scheduler.step(curr)

log_file.close()
writer.close()
