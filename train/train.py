# File: train/train.py
# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ===== Stdlib / third-party =====
import re, io, json, yaml, argparse, datetime, gc, subprocess, platform
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
from losses.focal_tversky import CombinedFocalTverskyLoss
from utils.plotting import add_prediction_legend
from yaml.representer import RepresenterError

# ===== Utilities =====
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

def update_confmat(confmat, preds, target, num_classes, ignore_index):
    if preds.is_cuda: preds = preds.cpu()
    if target.is_cuda: target = target.cpu()
    if preds.ndim == 3:
        preds  = preds.reshape(-1)
        target = target.reshape(-1)
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    if not torch.any(valid):
        return confmat
    t = target[valid].to(torch.int64)
    p = preds[valid].to(torch.int64)
    k = t * num_classes + p
    binc = torch.bincount(k, minlength=num_classes * num_classes)
    confmat += binc.view(num_classes, num_classes)
    return confmat

def state_dict_half_cpu(model: torch.nn.Module) -> dict:
    """Safe checkpoint: params/buffers -> CPU; float tensors cast to fp16."""
    sd = {}
    with torch.no_grad():
        for k, v in model.state_dict().items():
            t = v.detach().to('cpu', copy=True)
            if t.is_floating_point():
                t = t.to(torch.float16)
            sd[k] = t
    return sd

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

# Device early
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # fixed patch shapes → faster convs

# Resolve output directory
now_token = re.search(r"(\d{8}_\d{6})", os.path.basename(args.splits))
timestamp = now_token.group(1) if now_token else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = (args.out or cfg.get("output_dir", f"outputs/{timestamp}")).replace("{now}", timestamp)
os.makedirs(outdir, exist_ok=True)

print("[DEBUG] Training start:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("[INFO] Output dir:", outdir)

# ===== Load splits JSON (keys + stats) =====
with open(args.splits, "r") as f:
    splits_meta = json.load(f)

train_keys = list(splits_meta.get("train", []))
val_keys   = list(splits_meta.get("val", []))
test_keys  = list(splits_meta.get("test", []))  # not used here
patch_size = int(cfg.get("patch_size", 512))

norm_meta  = splits_meta.get("normalization", {})
class_meta = splits_meta.get("class_stats", {})

num_classes = int(cfg.get("num_classes", 13))
ignore_index = int(cfg.get("ignore_val", 255))
nodata_val   = cfg.get("nodata_val", None)
expected_C   = int(cfg.get("input_channels", 64))

# class weights from JSON → tensor on device once
cw_values = class_meta.get("weights", {}).get("values", {})
class_weights = torch.tensor(
    [float(cw_values.get(str(i), 1.0)) for i in range(num_classes)],
    dtype=torch.float32,
    device=device
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
                 nodata_val=nodata_val, ignore_index=ignore_index,
                 patch_size=512, is_train=True):
        self.keys = list(keys)
        self.inputs_idx = inputs_idx
        self.labels_idx = labels_idx
        self.mean_t = mean_t.float()         # [C,1,1]
        self.std_t  = std_t.float()          # [C,1,1]
        self.expected_channels = expected_channels
        self.transform = transform
        self.nodata_val = nodata_val
        self.ignore_index = ignore_index
        self.patch_size = int(patch_size)
        self.is_train = bool(is_train)
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

        # mask nodata/NaN → fill with per-channel mean so it normalizes to ~0
        mask = ~np.isfinite(x)
        if nod is not None:
            mask |= (x == nod)
        if mask.any():
            fill = self.mean_t.view(-1, 1, 1).numpy().astype(np.float32)  # [C,1,1]
            x = np.where(mask, fill, x)

        # normalize
        t = torch.from_numpy(x)
        t = (t - self.mean_t) / (self.std_t + 1e-6)
        return t  # [C,H,W] tensor

    def _read_label(self, path: str) -> torch.Tensor:
        with rasterio.open(path) as src:
            y = src.read(1)
            nod = src.nodata
        y = y.astype(np.int64)
        if nod is not None:
            y[y == nod] = self.ignore_index
        return torch.from_numpy(y)  # [H,W]

    def _pad_to_min_size(self, x: torch.Tensor, y: torch.Tensor, ps: int):
        _, H, W = x.shape
        pad_h = max(0, ps - H)
        pad_w = max(0, ps - W)
        if pad_h == 0 and pad_w == 0:
            return x, y

        top  = pad_h // 2
        bot  = pad_h - top
        left = pad_w // 2
        right= pad_w - left

        # inputs already normalized → 0 is neutral
        x = torch.nn.functional.pad(x, (left, right, top, bot), mode="constant", value=0.0)
        y = torch.nn.functional.pad(y, (left, right, top, bot), mode="constant", value=self.ignore_index)
        return x, y

    def _crop_pair(self, x: torch.Tensor, y: torch.Tensor, ps: int):
        x, y = self._pad_to_min_size(x, y, ps)
        _, H, W = x.shape
        if H == ps and W == ps:
            return x, y
        if self.is_train:
            i = np.random.randint(0, H - ps + 1)
            j = np.random.randint(0, W - ps + 1)
        else:
            i = (H - ps) // 2
            j = (W - ps) // 2
        x = x[:, i:i+ps, j:j+ps]
        y = y[i:i+ps, j:j+ps]
        return x, y

    def __getitem__(self, idx):
        k = self.keys[idx]
        xin = self.inputs_idx[k]
        ylb = self.labels_idx[k]

        x = self._read_input(xin)   # [C,H,W]
        y = self._read_label(ylb)   # [H,W]

        x, y = self._crop_pair(x, y, self.patch_size)

        if self.transform is not None:
            x, y = self.transform(x, y)
        return x, y

# ===== Transforms, loaders =====
train_transform = RandomAugment(p=0.5)
val_transform   = None

batch_size = args.batch_size or int(cfg.get("batch_size", 16))
epochs     = args.epochs or int(cfg.get("training", {}).get("epochs", 100))
num_workers = max(0, min(int(args.workers), 4))  # cap to 4 to be safe

train_ds = KeysDataset(train_keys, inputs_idx, labels_idx, norm_mean_t, norm_std_t,
                       expected_channels=expected_C, transform=train_transform,
                       nodata_val=nodata_val, ignore_index=ignore_index,
                       patch_size=patch_size, is_train=True)

val_ds   = KeysDataset(val_keys, inputs_idx, labels_idx, norm_mean_t, norm_std_t,
                       expected_channels=expected_C, transform=val_transform,
                       nodata_val=nodata_val, ignore_index=ignore_index,
                       patch_size=patch_size, is_train=False)

def make_loader(ds, batch, shuffle, workers, is_val=False):
    kwargs = dict(
        dataset=ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=safe_collate,
        drop_last=not is_val,  # drop last for train to keep shapes stable
    )
    if workers > 0:
        kwargs.update(dict(prefetch_factor=1, persistent_workers=False))
    return DataLoader(**kwargs)

train_loader = make_loader(train_ds, batch_size, True,  num_workers, is_val=False)
val_loader   = make_loader(val_ds,   batch_size, False, min(num_workers, 2), is_val=True)

print(f"[INFO] Loaded: train batches={len(train_loader)}  val batches={len(val_loader)}")

# ===== TensorBoard =====
tb_cfg = cfg.get("tensorboard", {})
if tb_cfg.get("restart", True):
    restart_tensorboard(outdir, port=int(tb_cfg.get("port", 6006)))
writer = SummaryWriter(log_dir=outdir, flush_secs=30, max_queue=10)

# ===== Model / Optim / Loss =====
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

use_amp = bool(cfg["training"].get("use_amp", False))
scaler  = GradScaler(enabled=use_amp)

es_metric = str(cfg["training"].get("early_stopping_metric", "loss")).lower()
best_metric = float("inf") if es_metric == "loss" else -1.0
patience = int(cfg["training"].get("early_stopping_patience", 30))
no_improve = 0

# ===== Scheduler (read from training.* keys) =====
sch_name = str(cfg["training"].get("scheduler", "reduce_on_plateau"))
sch_pat  = int(cfg["training"].get("scheduler_patience", 10))
sch_fac  = float(cfg["training"].get("scheduler_factor", 0.5))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode="min" if es_metric == "loss" else "max",
    factor=sch_fac,
    patience=sch_pat
)

# ===== Run metadata / config logging (AFTER model exists) =====
def _count_params(m: torch.nn.Module):
    tot = sum(p.numel() for p in m.parameters())
    trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return int(tot), int(trn)

def _git_info():
    info = {"commit": "unknown", "dirty": False}
    try:
        c = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        s = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode()
        info["commit"], info["dirty"] = c, (len(s.strip()) > 0)
    except Exception:
        pass
    return info

_total, _trainable = _count_params(model)
# ------- helper to make everything YAML/JSON-safe -------
def _pyify(obj):
    import numpy as _np
    import torch as _torch
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        # keep small arrays readable, otherwise summarize
        return obj.tolist() if obj.size <= 32 else f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
    if isinstance(obj, _torch.Tensor):
        return obj.detach().cpu().tolist() if obj.numel() <= 32 else f"tensor(shape={tuple(obj.shape)}, dtype={obj.dtype})"
    if isinstance(obj, dict):
        return {str(k): _pyify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pyify(v) for v in obj]
    # last resort: string
    return str(obj)

# ------- build metadata dict (coerce versions to str explicitly) -------
run_meta = {
    "time": datetime.datetime.now().isoformat(timespec="seconds"),
    "device": str(device),
    "torch": str(torch.__version__),                           # <- force str
    "cuda": str(torch.version.cuda) if torch.version.cuda else "none",
    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    "python": str(platform.python_version()),
    "platform": str(platform.platform()),
    "git": _git_info(),  # your helper
    "data": {
        "num_classes": int(num_classes),
        "channels": int(expected_C),
        "patch_size": int(patch_size),
        "n_train": int(len(train_keys)),
        "n_val": int(len(val_keys)),
    },
    "model": cfg.get("model", {}),
    "training": {
        "epochs": int(epochs),
        "lr": float(cfg["training"].get("lr", 1e-4)),
        "weight_decay": float(cfg["training"].get("weight_decay", 0.01)),
        "use_amp": bool(cfg["training"].get("use_amp", False)),
        "early_stopping_metric": str(es_metric),
        "early_stopping_patience": int(cfg["training"].get("early_stopping_patience", 30)),
        "grad_clip": float(cfg.get("gradient_clipping", 1.0)),
        "scheduler": {"name": sch_name, "patience": sch_pat, "factor": sch_fac},
        "batch_size": int(batch_size),
        "workers": int(num_workers),
    },
    "loss": cfg.get("loss", {}),
    "splitting": cfg.get("splitting", {}),
    "class_weights_summary": {
        "min": float(class_weights.min().item()),
        "max": float(class_weights.max().item()),
        "mean": float(class_weights.mean().item()),
    },
    "model_params": {"total": int(_total), "trainable": int(_trainable)},
}

# ------- write & log (YAML → fallback to JSON text) -------
try:
    clean_meta = _pyify(run_meta)

    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(clean_meta, f, indent=2)

    try:
        tb_yaml = yaml.safe_dump(clean_meta, sort_keys=False, default_flow_style=False)
        writer.add_text("run/config", f"```yaml\n{tb_yaml}\n```", global_step=0)
    except RepresenterError:
        writer.add_text("run/config", "```json\n" + json.dumps(clean_meta, indent=2) + "\n```", global_step=0)

    # lightweight hparams (must be simple scalars/strings)
    hp = {
        "model.encoder": str(cfg.get("model", {}).get("encoder", "resnet34")),
        "training.lr": float(run_meta["training"]["lr"]),
        "training.weight_decay": float(run_meta["training"]["weight_decay"]),
        "training.scheduler": str(sch_name),
        "data.patch_size": int(patch_size),
        "data.channels": int(expected_C),
    }
    writer.add_hparams(hp, {"init/best_loss": 0.0, "init/best_f1": 0.0})
    writer.flush()
except Exception as e:
    print(f"[WARN] metadata logging failed: {e}")

# ===== Training loop =====
SAVE_EVERY = int(cfg.get("logging", {}).get("checkpoint_interval", 5))
VIS_EVERY  = int(cfg.get("logging", {}).get("eval_interval", 1))  # visualization cadence
clip_val   = float(cfg.get("gradient_clipping", 1.0))

log_f = open(os.path.join(outdir, "training_log.csv"), "a", buffering=1)
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

            # CE on valid pixels only, using pre-moved class_weights
            valid_mask = (yb != ignore_index) & (yb >= 0) & (yb < num_classes)
            if valid_mask.any():
                ce = nn.functional.cross_entropy(
                    logits.permute(0,2,3,1)[valid_mask],
                    yb[valid_mask],
                    weight=class_weights,
                    ignore_index=ignore_index
                )
            else:
                ce = torch.tensor(0.0, device=device)

            loss = ft_loss(logits, yb) + ce

        scaler.scale(loss).backward()
        # unscale before clipping so the norm is correct
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        scaler.step(opt)
        scaler.update()

        run_loss += float(loss.item())
        # cleanup
        del xb, yb, logits, loss, ce, valid_mask

    avg_loss = run_loss / max(1, len(train_loader))

    # ===== Validation (streaming, O(1) memory) =====
    model.eval()
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    correct, total = 0, 0

    with torch.inference_mode():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(1)

            valid = (yb != ignore_index) & (yb >= 0) & (yb < num_classes)
            correct += (pred[valid] == yb[valid]).sum().item()
            total   += valid.sum().item()

            confmat = update_confmat(confmat, pred.detach(), yb.detach(), num_classes, ignore_index)

            del xb, yb, logits, pred

    # derive mIoU and F1 from confmat (CPU)
    tp = confmat.diag().to(torch.float64)
    fp = confmat.sum(0).to(torch.float64) - tp
    fn = confmat.sum(1).to(torch.float64) - tp
    den = tp + fp + fn + 1e-9
    iou_per_class = tp / den
    miou = torch.nanmean(iou_per_class).item()

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1_per_class = (2 * precision * recall) / (precision + recall + 1e-9)
    f1 = torch.nanmean(f1_per_class).item()

    acc = correct / max(1, total)

    # Logging
    lr = opt.param_groups[0]["lr"]
    print(f"Epoch {epoch}: loss={avg_loss:.4f}  acc={acc:.4f}  mIoU={miou:.4f}  F1={f1:.4f}  lr={lr:.2e}")
    log_f.write(f"{epoch},{avg_loss:.6f},{acc:.6f},{miou:.6f},{f1:.6f},{lr:.6e}\n"); log_f.flush()

    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/val", acc, epoch)
    writer.add_scalar("mIoU/val", miou, epoch)
    writer.add_scalar("F1/val", f1, epoch)
    writer.add_scalar("LR", lr, epoch)

    # ===== Scheduler + early stopping =====
    score = avg_loss if es_metric == "loss" else f1
    scheduler.step(score)

    improved = (score < best_metric) if es_metric == "loss" else (score > best_metric)
    should_save = improved or (epoch % SAVE_EVERY == 0)

    # don't spam TB with images; space them out and skip when saving
    vis_stride = max(VIS_EVERY, 5)
    should_visual_log = (epoch % vis_stride == 0) and not should_save

    if should_save:
        if improved:
            new_best = score  # avg_loss or f1, depending on es_metric
            with open(os.path.join(outdir, "best_epoch.txt"), "w") as f:
                f.write(f"{epoch},{new_best:.6f}\n")
            best_metric = new_best
            no_improve = 0
        else:
            no_improve += 1

        fname = ("best_model_ep%03d.pt" % epoch) if improved else ("model_ep%03d.pt" % epoch)
        try:
            state = state_dict_half_cpu(model)
            tmp_path = os.path.join(outdir, fname + ".tmp")
            final_path = os.path.join(outdir, fname)
            torch.save(state, tmp_path, _use_new_zipfile_serialization=False)
            os.replace(tmp_path, final_path)
            del state
        except Exception as e:
            print(f"[WARN] checkpoint save failed: {e}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    # ===== Visualize ONLY when not saving this epoch =====
    if should_visual_log:
        fig = buf = img = None
        try:
            model.eval()
            n_show = min(3, len(val_ds))
            fig, axs = plt.subplots(n_show, 2, figsize=(8, 3*n_show), dpi=120)
            if n_show == 1:
                axs = np.array([[axs[0], axs[1]]])
            for i in range(n_show):
                x, y = val_ds[i]
                with torch.inference_mode():
                    pred = model(x.unsqueeze(0).to(device)).argmax(1).squeeze(0).cpu()
                axs[i,0].imshow(y.numpy(), cmap="tab20"); axs[i,0].set_title("Label"); axs[i,0].axis("off")
                axs[i,1].imshow(pred.numpy(), cmap="tab20"); axs[i,1].set_title("Prediction"); axs[i,1].axis("off")
                del x, y, pred
            add_prediction_legend(axs[-1,1], num_classes, cfg.get("label_names", {}))
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig); fig = None
            buf.seek(0)
            img = Image.open(buf)
            writer.add_image("Val/Label_vs_Pred", to_tensor(img), global_step=epoch)
            writer.flush()
        except Exception as e:
            print(f"[WARN] TensorBoard visualization failed: {e}")
        finally:
            try:
                if img is not None: img.close()
            except Exception: pass
            try:
                if buf is not None: buf.close()
            except Exception: pass
            del img, buf
            torch.cuda.empty_cache(); gc.collect()

    torch.cuda.empty_cache(); gc.collect()

# ===== Cleanup =====
log_f.close()
writer.close()
print("[OK] Training finished.")
