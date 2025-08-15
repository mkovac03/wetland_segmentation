# File: train/train.py
# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ===== Stdlib / third-party =====
import re, io, json, yaml, argparse, datetime, hashlib, gc, subprocess
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor

# ===== Project modules =====
from data.transform import RandomAugment
from models.resunet_vit import ResNetUNetViT
from train.metrics import compute_miou, compute_f1
from losses.focal_tversky import CombinedFocalTverskyLoss
from utils.plotting import add_prediction_legend

# ===== Utilities (optional GPU/RAM logging; safe if NVML missing) =====
def restart_tensorboard(logdir, port=6006):
    try:
        out = subprocess.check_output(["pgrep", "-f", f"tensorboard.*{logdir}"])
        for pid in out.decode().strip().split("\n"):
            if pid:
                subprocess.run(["kill", "-9", pid], check=False)
    except subprocess.CalledProcessError:
        pass
    try:
        subprocess.Popen([
            "tensorboard",
            f"--logdir={logdir}",
            f"--port={port}",
            "--host=127.0.0.1",
            "--reload_interval=5",
            "--load_fast=false"
        ])
        print(f"[INFO] TensorBoard at http://localhost:{port}")
    except Exception as e:
        print(f"[WARN] TensorBoard start failed: {e}")

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("safe_collate: all samples failed.")
    return default_collate(batch)

# ===== Argparse / Config =====
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/config.yaml")
ap.add_argument("--splits", required=True, help="verified splits JSON from tools/split_data.py")
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--batch-size", type=int, default=None)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--out", default=None)
args = ap.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# Resolve output directory
now_token = re.search(r"(\d{8}_\d{6})", os.path.basename(args.splits))
timestamp = now_token.group(1) if now_token else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = args.out or cfg.get("output_dir", f"outputs/{timestamp}")
outdir = outdir.replace("{now}", timestamp)
os.makedirs(outdir, exist_ok=True)

print("[DEBUG] Training start:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("[INFO] Output dir:", outdir)

# ===== Load splits JSON (keys + stats) =====
with open(args.splits, "r") as f:
    splits_meta = json.load(f)

train_keys = list(splits_meta.get("train", []))
val_keys   = list(splits_meta.get("val", []))
test_keys  = list(splits_meta.get("test", []))  # not used here, but available

norm_meta  = splits_meta.get("normalization", {})
class_meta = splits_meta.get("class_stats", {})

num_classes = int(cfg.get("num_classes", 13))
ignore_index = int(cfg.get("ignore_val", 255))
nodata_val   = cfg.get("nodata_val", None)
expected_C   = int(cfg.get("input_channels", 64))

# class weights from JSON
cw_values = class_meta.get("weights", {}).get("values", {})
# convert dict with string keys → ordered list
class_weights = torch.tensor(
    [float(cw_values.get(str(i), 1.0)) for i in range(num_classes)],
    dtype=torch.float32
)

# normalization tensors
means = np.array(norm_meta.get("mean", [0.0]*expected_C), dtype=np.float32)
stds  = np.array(norm_meta.get("std",  [1.0]*expected_C), dtype=np.float32)
if means.size != expected_C or stds.size != expected_C:
    print(f"[WARN] normalization size mismatch; expected {expected_C}, got {means.size}/{stds.size}")
norm_mean_t = torch.from_numpy(means)[:, None, None]
norm_std_t  = torch.from_numpy(stds)[:, None, None]

# ===== Index inputs/labels under <input_dir> =====
def extract_key_from_name(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    nums = re.findall(r'\d+', stem)
    if not nums: return None
    zones = [n for n in nums if re.fullmatch(r'32[67]\d{2}', n)]
    if not zones: return None
    zone = zones[-1]
    tail_ids = re.findall(r'_(\d+)', stem)
    tile = tail_ids[-1] if tail_ids else nums[-1]
    if tile == zone and len(nums) >= 2:
        tile = nums[-2]
    if tile == zone:
        return None
    return f"{zone}_{tile}"

def index_by_key(root: str, exts: Tuple[str, ...]) -> Dict[str, str]:
    idx = {}
    if not root or not os.path.isdir(root): return idx
    exts = tuple(e.lower() for e in exts)
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(exts):
                k = extract_key_from_name(fn)
                if k:
                    idx[k] = os.path.join(dp, fn)
    return idx

base = cfg.get("input_dir")
inputs_root = os.path.join(base, "inputs")
labels_root = os.path.join(base, "labels")
inputs_idx  = index_by_key(inputs_root, ("tif","tiff","vrt","npy"))
labels_idx  = index_by_key(labels_root, ("tif","tiff","vrt"))

def resolve_overlap(keys: List[str]) -> List[str]:
    return [k for k in keys if (k in inputs_idx and k in labels_idx)]

train_keys = resolve_overlap(train_keys)
val_keys   = resolve_overlap(val_keys)

print(f"[INFO] Resolved: train={len(train_keys)}  val={len(val_keys)}")

if len(train_keys) == 0 or len(val_keys) == 0:
    raise RuntimeError("No overlapping tiles between inputs and labels for train/val.")

# ===== Dataset =====
class KeysDataset(Dataset):
    def __init__(self, keys, inputs_idx, labels_idx, mean_t, std_t,
                 expected_channels=expected_C, transform=None,
                 nodata_val=nodata_val, ignore_index=ignore_index):
        self.keys = list(keys)
        self.inputs_idx = inputs_idx
        self.labels_idx = labels_idx
        self.mean_t = mean_t.float()
        self.std_t  = std_t.float()
        self.expected_channels = expected_channels
        self.transform = transform
        self.nodata_val = nodata_val
        self.ignore_index = ignore_index
        # keep file paths for TB previews
        self.file_list = [self.inputs_idx[k] for k in self.keys]

    def __len__(self): return len(self.keys)

    def _read_input(self, path: str) -> torch.Tensor:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".tif", ".tiff", ".vrt"):
            with rasterio.open(path) as src:
                arr = src.read()  # [C,H,W]
                nod = src.nodata
        elif ext == ".npy":
            arr = np.load(path)
            nod = self.nodata_val
        else:
            raise ValueError(f"Unsupported input ext: {ext}")
        if arr.ndim != 3:
            raise ValueError(f"Input must be [C,H,W], got {arr.shape} at {path}")
        if arr.shape[0] < self.expected_channels:
            raise ValueError(f"Channels<{self.expected_channels} in {path}: {arr.shape}")
        x = arr[:self.expected_channels].astype(np.float32)

        # mask nodata/NaN → fill with channel mean so it normalizes to ~0
        mask = ~np.isfinite(x)
        if nod is not None:
            mask |= (x == nod)
        if mask.any():
            # broadcast per-channel means
            ch_means = self.mean_t.numpy().astype(np.float32)
            for c in range(x.shape[0]):
                x[c][mask[c]] = ch_means[c]
        t = torch.from_numpy(x)
        # normalize
        t = (t - self.mean_t) / (self.std_t + 1e-6)
        return t

    def _read_label(self, path: str) -> torch.Tensor:
        with rasterio.open(path) as src:
            y = src.read(1)
            nod = src.nodata
        y = y.astype(np.int64)
        if nod is not None:
            y[y == nod] = self.ignore_index
        return torch.from_numpy(y)

    def __getitem__(self, idx):
        k = self.keys[idx]
        xin = self.inputs_idx[k]
        ylb = self.labels_idx[k]
        x = self._read_input(xin)
        y = self._read_label(ylb)
        if self.transform is not None:
            # Expect transform to take (x,y) tensors and return transformed (x,y)
            x, y = self.transform(x, y)
        return x, y

# ===== Transforms, loaders =====
train_transform = RandomAugment(p=0.5)
val_transform   = None

batch_size = args.batch_size or int(cfg.get("batch_size", 16))
epochs     = args.epochs or int(cfg.get("training", {}).get("epochs", 100))
num_workers= int(args.workers)

train_ds = KeysDataset(train_keys, inputs_idx, labels_idx, norm_mean_t, norm_std_t,
                       expected_channels=expected_C, transform=train_transform,
                       nodata_val=nodata_val, ignore_index=ignore_index)
val_ds   = KeysDataset(val_keys, inputs_idx, labels_idx, norm_mean_t, norm_std_t,
                       expected_channels=expected_C, transform=val_transform,
                       nodata_val=nodata_val, ignore_index=ignore_index)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, prefetch_factor=2,
    persistent_workers=(num_workers > 0), collate_fn=safe_collate
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True, prefetch_factor=2,
    persistent_workers=(num_workers > 0), collate_fn=safe_collate
)

print(f"[INFO] Loaded: train batches={len(train_loader)}  val batches={len(val_loader)}")

# ===== TensorBoard =====
tb_cfg = cfg.get("tensorboard", {})
if tb_cfg.get("restart", True):
    restart_tensorboard(outdir, port=int(tb_cfg.get("port", 6006)))
writer = SummaryWriter(log_dir=outdir)

# ===== Model / Optim / Loss =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ResNetUNetViT(cfg).to(device)

opt = optim.AdamW(model.parameters(),
                  lr=float(cfg["training"].get("lr", 1e-4)),
                  weight_decay=float(cfg["training"].get("weight_decay", 0.01)))

ft_loss = CombinedFocalTverskyLoss(
    num_classes=num_classes,
    focal_alpha=float(cfg["loss"]["focal"]["alpha"]),
    focal_gamma=float(cfg["loss"]["focal"]["gamma"]),
    boundary_weight=float(cfg["loss"]["focal"]["boundary_weight"]),
    tversky_alpha=float(cfg["loss"]["tversky"]["alpha"]),
    tversky_beta=float(cfg["loss"]["tversky"]["beta"]),
    ignore_index=ignore_index
)

ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=ignore_index)

use_amp = bool(cfg["training"].get("use_amp", False))
scaler  = GradScaler(enabled=use_amp)

es_metric = str(cfg["training"].get("early_stopping_metric", "loss")).lower()
best_metric = float("inf") if es_metric == "loss" else -1.0
patience = int(cfg["training"].get("early_stopping_patience", 30))
no_improve = 0

sch_cfg = cfg.get("scheduler", {"mode": "reduce_on_plateau", "patience": 10})
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode="min" if es_metric == "loss" else "max",
    factor=float(sch_cfg.get("factor", 0.5)),
    patience=int(sch_cfg.get("patience", 10))
)

# ===== Training loop =====
SAVE_EVERY = int(cfg.get("logging", {}).get("checkpoint_interval", 5))
VIS_EVERY  = int(cfg.get("logging", {}).get("eval_interval", 1))  # we use it as visual cadence
clip_val   = float(cfg.get("gradient_clipping", 1.0))

log_f = open(os.path.join(outdir, "training_log.csv"), "a")
if log_f.tell() == 0:
    log_f.write("epoch,loss,acc,miou,f1,lr\n"); log_f.flush()

for epoch in range(1, epochs + 1):
    model.train()
    run_loss = 0.0

    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(xb)
            # combine losses: Focal+Tversky (full) + CE on valid pixels
            valid_mask = (yb != ignore_index) & (yb >= 0) & (yb < num_classes)
            if valid_mask.any():
                ce = nn.functional.cross_entropy(
                    logits.permute(0,2,3,1)[valid_mask],
                    yb[valid_mask],
                    weight=class_weights.to(device),
                    ignore_index=ignore_index
                )
            else:
                ce = torch.tensor(0.0, device=device)
            loss = ft_loss(logits, yb) + ce

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        scaler.step(opt)
        scaler.update()

        run_loss += float(loss.item())

        del xb, yb, logits, loss, ce, valid_mask
        torch.cuda.empty_cache(); gc.collect()

    avg_loss = run_loss / max(1, len(train_loader))

    # ===== Validation =====
    model.eval()
    correct, total = 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(1)
            preds_all.append(pred.cpu())
            labels_all.append(yb.cpu())
            correct += (pred == yb).sum().item()
            total   += yb.numel()
            del xb, yb, logits, pred
    acc  = correct / max(1, total)
    miou = compute_miou(preds_all, labels_all, num_classes)
    f1   = compute_f1(preds_all, labels_all, num_classes)
    del preds_all, labels_all

    # Logging
    lr = opt.param_groups[0]["lr"]
    print(f"Epoch {epoch}: loss={avg_loss:.4f}  acc={acc:.4f}  mIoU={miou:.4f}  F1={f1:.4f}  lr={lr:.2e}")
    log_f.write(f"{epoch},{avg_loss:.6f},{acc:.6f},{miou:.6f},{f1:.6f},{lr:.6e}\n"); log_f.flush()

    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/val", acc, epoch)
    writer.add_scalar("mIoU/val", miou, epoch)
    writer.add_scalar("F1/val", f1, epoch)
    writer.add_scalar("LR", lr, epoch)

    # Scheduler + early stopping
    score = avg_loss if es_metric == "loss" else f1
    scheduler.step(score if es_metric == "loss" else f1)

    improved = (score < best_metric) if es_metric == "loss" else (f1 > best_metric)
    should_save = improved or (epoch % SAVE_EVERY == 0)
    should_visual_log = ((epoch % VIS_EVERY == 0) or epoch == 1) and not should_save

    if should_save:
        best_metric = score if es_metric == "loss" else (f1 if improved else best_metric)
        no_improve = 0 if improved else (no_improve + 1)

        state = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in model.state_dict().items()}
        fname = ("best_model_ep%03d.pt" % epoch) if improved else ("model_ep%03d.pt" % epoch)
        torch.save(state, os.path.join(outdir, fname), _use_new_zipfile_serialization=False)

        if improved:
            with open(os.path.join(outdir, "best_epoch.txt"), "w") as bf:
                bf.write(f"{epoch},{best_metric:.6f}\n")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    # ===== Visualize ONLY when not saving this epoch =====
    if should_visual_log:
        try:
            model.eval()
            n_show = min(3, len(val_ds))
            fig, axs = plt.subplots(n_show, 2, figsize=(8, 3*n_show), dpi=120)
            if n_show == 1:
                axs = np.array([[axs[0], axs[1]]])
            for i in range(n_show):
                x, y = val_ds[i]
                with torch.no_grad():
                    pred = model(x.unsqueeze(0).to(device)).argmax(1).squeeze(0).cpu()
                axs[i,0].imshow(y.numpy(), cmap="tab20"); axs[i,0].set_title("Label"); axs[i,0].axis("off")
                axs[i,1].imshow(pred.numpy(), cmap="tab20"); axs[i,1].set_title("Prediction"); axs[i,1].axis("off")
            add_prediction_legend(axs[-1,1], num_classes, cfg.get("label_names", {}))
            plt.tight_layout()
            buf = io.BytesIO(); plt.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
            img = Image.open(buf); writer.add_image("Val/Label_vs_Pred", to_tensor(img), global_step=epoch)
            writer.flush()
        except Exception as e:
            print(f"[WARN] TensorBoard visualization failed: {e}")

# ===== Cleanup =====
log_f.close()
writer.close()
print("[OK] Training finished.")
