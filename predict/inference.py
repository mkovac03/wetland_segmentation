# -*- coding: utf-8 -*-
"""
Unified inference:
- Robust weight resolution (best_epoch.txt or best metric in filename)
- Input discovery: {inference.input_dir | processed_dir} with {now} -> --timestamp
- Auto-detects .tif/.tiff/.npy tiles
- Skip all-NoData tiles; propagate label=255; never predict 255
- Saves GeoTIFF preds for GeoTIFF inputs, and *_pred.npy for .npy inputs
"""

import os
import re
import sys
import glob
import argparse
import collections
import numpy as np
import torch
import yaml
import zlib
from tqdm import tqdm
from scipy.special import softmax

# Optional raster IO (only needed for GeoTIFF inputs/outputs)
import rasterio
from rasterio.transform import Affine

# -------------------------------------------------
# Ensure repo imports work no matter where we run from
# -------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from models.resunet_vit import ResNetUNetViT  # noqa: E402

# -------------------------------------------------
# Load config
# -------------------------------------------------
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

GEN = config
MODEL_CFG = config.get("model", {})
INFER_CFG = config.get("inference", {})

# Core data settings
NUM_CLASSES = int(GEN.get("num_classes", 13))
INPUT_CHANNELS = int(GEN.get("input_channels", 22))

# Inference flags
LABEL_IN_FIRST_BAND = bool(INFER_CFG.get("label_in_first_band", True))
NODATA_LABEL = int(INFER_CFG.get("nodata_label", 255))
NODATA_VALUE_INPUT = INFER_CFG.get("nodata_value_input", -32768)
SAVE_PROBS = bool(INFER_CFG.get("save_probabilities", False))
BAND_OFFSET = int(INFER_CFG.get("band_offset", 0))


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def as_abs(path_like: str) -> str:
    return path_like if os.path.isabs(path_like) else os.path.join(REPO_ROOT, path_like)

def resolve_input_root(timestamp: str) -> str:
    # 1) inference.input_dir
    cand = INFER_CFG.get("input_dir")
    if cand:
        cand = cand.replace("{now}", timestamp)
        cand_abs = as_abs(cand)
        if os.path.isdir(cand_abs):
            return cand_abs
    # 2) processed_dir from general config
    cand = GEN.get("processed_dir", "data/processed/{now}").replace("{now}", timestamp)
    cand_abs = as_abs(cand)
    if os.path.isdir(cand_abs):
        return cand_abs
    # 3) common default
    cand_abs = as_abs(os.path.join("data", "processed", timestamp))
    if os.path.isdir(cand_abs):
        return cand_abs
    # 4) outputs/<ts>/inputs
    cand_abs = os.path.join(resolve_outputs_dir(timestamp), "inputs")
    if os.path.isdir(cand_abs):
        return cand_abs
    raise FileNotFoundError(
        f"Could not resolve input dir. Tried inference.input_dir, processed_dir, "
        f"data/processed/{timestamp}, and outputs/{timestamp}/inputs"
    )

def resolve_outputs_dir(timestamp: str) -> str:
    tmpl = GEN.get("output_dir", os.path.join(REPO_ROOT, "outputs", "{now}"))
    base = tmpl.replace("{now}", timestamp)
    return base if os.path.isabs(base) else os.path.join(REPO_ROOT, base)

def parse_best_epoch_txt(run_dir: str):
    p = os.path.join(run_dir, "best_epoch.txt")
    if os.path.isfile(p):
        try:
            with open(p, "r") as f:
                text = f.read().strip()
            m = re.search(r"(\d+)", text)
            if m:
                return int(m.group(1))
        except Exception:
            pass
    return None

def find_best_checkpoint(run_dir: str):
    # 1) If best_epoch.txt exists, try to match epoch in filenames
    weight_glob = os.path.join(run_dir, "*.pt")
    candidates = glob.glob(weight_glob)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints (*.pt) in {run_dir}")

    best_epoch = parse_best_epoch_txt(run_dir)
    if best_epoch is not None:
        for c in candidates:
            name = os.path.basename(c)
            if re.search(rf"(epoch[=_-]{best_epoch}\b)|(^{best_epoch}[,_-])", name):
                return c
        for c in candidates:
            nums = re.findall(r"\d+", os.path.basename(c))
            if nums and int(nums[0]) == best_epoch:
                return c

    # 2) Otherwise, choose by best metric parsed from filename (f1 or miou)
    scored = []
    for c in candidates:
        name = os.path.basename(c)
        m_f1 = re.search(r"f1[=_-]?([0-9]*\.?[0-9]+)", name, re.IGNORECASE)
        m_iou = re.search(r"(miou|iou)[=_-]?([0-9]*\.?[0-9]+)", name, re.IGNORECASE)
        metric = None
        if m_f1:
            metric = float(m_f1.group(1))
        elif m_iou:
            metric = float(m_iou.group(2))
        else:
            parts = re.findall(r"[0-9]*\.?[0-9]+", name)
            if len(parts) >= 2:
                try:
                    metric = float(parts[1])
                except Exception:
                    pass
        if metric is not None:
            scored.append((metric, c))
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(f"No suitable checkpoint found in {run_dir}")

def load_model(ckpt_path, device="cuda"):
    model = ResNetUNetViT(config).to(device)
    model.eval()

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    # ---- Diagnostics: print critical mismatches ----
    crit_missing = [k for k in missing if ("conv1.weight" in k or "classifier" in k or "seg_head" in k or "final" in k)]
    crit_unexpected = [k for k in unexpected if ("conv1.weight" in k or "classifier" in k or "seg_head" in k or "final" in k)]
    if missing or unexpected:
        print(f"[WARN] load_state_dict mismatches → missing: {len(missing)}, unexpected: {len(unexpected)}")
        if crit_missing:
            print("[WARN] missing (critical):", crit_missing[:6], "...")
        if crit_unexpected:
            print("[WARN] unexpected (critical):", crit_unexpected[:6], "...")

    # ---- Channel count sanity check ----
    in_ch_cfg = int(GEN.get("input_channels", 22))
    first_conv_in = None
    first_conv_wshape = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            first_conv_in = m.in_channels
            w = getattr(m, "weight", None)
            first_conv_wshape = tuple(w.shape) if isinstance(w, torch.Tensor) else None
            break
    if first_conv_in is not None:
        print(f"[CHECK] First Conv2d expects in_channels={first_conv_in} (weight shape={first_conv_wshape}), "
              f"config.input_channels={in_ch_cfg}")
        if first_conv_in != in_ch_cfg:
            print("[ERROR] Input channel mismatch. Your config and checkpoint disagree. "
                  "Either set config['input_channels'] to the checkpoint’s expected value or use matching weights.")

    # ---- Class count sanity check ----
    last_out = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_out = m.out_channels
    if last_out is not None:
        print(f"[CHECK] Final logits out_channels={last_out}, config.num_classes={GEN.get('num_classes')}")

    return model


def discover_files(input_root: str):
    # Prefer GeoTIFFs; then .npy
    for ext in ("tif", "tiff"):
        pat = os.path.join(input_root, f"**/*.{ext}")
        files = [f for f in glob.glob(pat, recursive=True)
                 if "_pred.tif" not in f and "_probs.npz" not in f]
        if files:
            return sorted(files), ext
    # .npy → only *_img.npy
    pat = os.path.join(input_root, "**/*_img.npy")
    npy_files = sorted(glob.glob(pat, recursive=True))
    if npy_files:
        return npy_files, "npy"
    return [], None

# ---------- IO for GeoTIFF ----------
def read_raster(path: str):
    with rasterio.open(path) as src:
        img = src.read()  # (C,H,W)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    return img, profile, transform, crs

def write_raster(path: str, array: np.ndarray, profile: dict, transform: Affine):
    prof = profile.copy()
    prof.update({
        "count": 1,
        "dtype": rasterio.uint8,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": min(256, array.shape[1]),
        "blockysize": min(256, array.shape[0]),
        "transform": transform,
        "nodata": 255,
    })
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array.astype(np.uint8), 1)

# ---------- IO for NPY ----------
def read_npy_pair(img_path: str):
    # Expect shape (C,H,W)
    img = np.load(img_path)
    if img.dtype != np.float32:
        img = img.astype(np.float32, copy=False)
    lbl_path = img_path.replace("_img.npy", "_lbl.npy")
    label = np.load(lbl_path).astype(np.uint8, copy=False) if os.path.isfile(lbl_path) else None
    return img, label

def write_npy_pred(img_path: str, pred: np.ndarray, out_dir: str | None = None):
    name = os.path.basename(img_path).replace("_img.npy", "_pred.npy")
    out = os.path.join(out_dir or os.path.dirname(img_path), name)
    np.save(out, pred.astype(np.uint8))
    return out

# -------------------------------------------------
# Core inference
# -------------------------------------------------
def is_all_nodata(x: np.ndarray, nodata_val) -> bool:
    if x.ndim != 3:
        return False
    all_neg = np.all(x == nodata_val)              # true sentinel everywhere
    all_255 = np.all(np.all(x == 255, axis=0))     # every pixel has ALL chans==255
    return all_neg or all_255


def predict_tile(model, x: np.ndarray, device: str = "cuda",
                 debias_vec: np.ndarray | None = None,
                 temperature: float = 1.0):
    with torch.no_grad():
        xt = torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,C,H,W)
        logits = model(xt)  # (1,K,H,W)

        if debias_vec is not None:
            if not torch.is_tensor(debias_vec):
                debias_vec = torch.tensor(debias_vec, dtype=logits.dtype, device=logits.device)
            logits = logits - debias_vec.view(1, -1, 1, 1)

        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=1).float()
        pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        probs_np = probs.squeeze(0).cpu().numpy()
    return pred, probs_np

def predict_tile_sliding(model, x, patch_size, stride, device="cuda",
                         debias_vec=None, temperature=1.0):
    """
    x: (C,H,W) float32, already fixed for NoData and channel count
    Returns: pred (H,W) uint8, probs (K,H,W) float32
    """
    C, H, W = x.shape
    K = int(GEN.get("num_classes", 13))

    logit_accum = np.zeros((K, H, W), dtype=np.float32)
    weight_map  = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, H, stride):
            for x0 in range(0, W, stride):
                y1, x1 = y, x0
                y2, x2 = min(y1 + patch_size, H), min(x1 + patch_size, W)
                dy, dx = y2 - y1, x2 - x1

                patch = x[:, y1:y2, x1:x2]
                pad_bottom = patch_size - dy
                pad_right  = patch_size - dx
                if pad_bottom or pad_right:
                    patch = np.pad(patch, ((0,0),(0,pad_bottom),(0,pad_right)), mode="edge")

                xt = torch.from_numpy(patch).unsqueeze(0).to(device=device, dtype=torch.float32)
                logits = model(xt).squeeze(0)  # (K, PS, PS)

                if debias_vec is not None:
                    dv = torch.tensor(debias_vec, dtype=logits.dtype, device=logits.device).view(-1, 1, 1)
                    logits = logits - dv
                if temperature != 1.0 and temperature > 0:
                    logits = logits / temperature

                logits_np = logits[:, :dy, :dx].cpu().numpy()
                logit_accum[:, y1:y2, x1:x2] += logits_np
                weight_map[y1:y2, x1:x2] += 1.0

    weight_map = np.clip(weight_map, 1e-6, None)
    avg_logits = logit_accum / weight_map
    probs = softmax(avg_logits, axis=0).astype(np.float32)
    pred  = np.argmax(probs, axis=0).astype(np.uint8)
    return pred, probs


def load_x_from_path(path, mode):
    """Loads one tile (float32), applies our NoData hot-fixes and optional label extraction."""
    if mode in ("tif", "tiff"):
        img, profile, transform, crs = read_raster(path)
        label_band = None
        if LABEL_IN_FIRST_BAND:
            if img.shape[0] < (1 + BAND_OFFSET + INPUT_CHANNELS):
                raise ValueError(f"Expected >= {1 + BAND_OFFSET + INPUT_CHANNELS} bands, got {img.shape[0]} for {path}")
            label_band = img[0].copy()
            x = img[1 + BAND_OFFSET: 1 + BAND_OFFSET + INPUT_CHANNELS].astype(np.float32, copy=False)
        else:
            if img.shape[0] < (BAND_OFFSET + INPUT_CHANNELS):
                raise ValueError(f"Expected >= {BAND_OFFSET + INPUT_CHANNELS} bands, got {img.shape[0]} for {path}")
            x = img[BAND_OFFSET: BAND_OFFSET + INPUT_CHANNELS].astype(np.float32, copy=False)
        # hot-fix:
        if (x == NODATA_VALUE_INPUT).any():
            x = x.copy(); x[x == NODATA_VALUE_INPUT] = 0.0
        mask_px_all255 = np.all(x == 255.0, axis=0)
        if mask_px_all255.any():
            if not x.flags.writeable: x = x.copy()
            x[:, mask_px_all255] = 0.0
        meta = {"profile": profile, "transform": transform, "label_band": label_band}
        return x, meta
    else:
        x, label_band = read_npy_pair(path)  # float32 already
        if x.shape[0] != INPUT_CHANNELS:
            if LABEL_IN_FIRST_BAND and x.shape[0] == INPUT_CHANNELS + 1:
                label_band = x[0].copy()
                x = x[1:1+INPUT_CHANNELS]
            else:
                x = x[:INPUT_CHANNELS]
        # hot-fix:
        if (x == NODATA_VALUE_INPUT).any():
            x = x.copy(); x[x == NODATA_VALUE_INPUT] = 0.0
        mask_px_all255 = np.all(x == 255.0, axis=0)
        if mask_px_all255.any():
            if not x.flags.writeable: x = x.copy()
            x[:, mask_px_all255] = 0.0
        meta = {"profile": None, "transform": None, "label_band": label_band}
        return x, meta

def bn_adapt_update(model, files, mode, device="cuda", n_tiles=128, scale32768=False):
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = 0.1
    with torch.no_grad():
        for pth in files[:max(1, n_tiles)]:
            x, _ = load_x_from_path(pth, mode)
            if scale32768: x = x / 32768.0
            xt = torch.from_numpy(x).unsqueeze(0).to(device, dtype=torch.float32)
            _ = model(xt)
    model.eval()


def compute_calib_debias_vec(model, files, mode, device="cuda", n_tiles=128, scale32768=False):
    """Average per-class logits over N tiles to form a debias vector."""
    model.eval()
    sum_vec = None
    count = 0
    with torch.no_grad():
        for pth in files[:max(1, n_tiles)]:
            x, _ = load_x_from_path(pth, mode)
            if scale32768: x = x / 32768.0
            xt = torch.from_numpy(x).unsqueeze(0).to(device, dtype=torch.float32)
            logits = model(xt).squeeze(0)  # (K,H,W)
            vec = logits.mean(dim=(1, 2))  # (K,)
            sum_vec = vec if sum_vec is None else (sum_vec + vec)
            count += 1
    return (sum_vec / max(1, count)).detach().cpu().numpy()


def run_inference_on_files(model, files, mode, out_dir, device="cuda",
                           limit=0, scale32768=False, debug_zero=False,
                           debias_zero=False, debias_calib=0, bn_adapt_tiles=0,
                           temperature=1.0, debug_fingerprint=False,
                           sliding=False, patch_size=None, stride=None):
    os.makedirs(out_dir, exist_ok=True)
    printed_stats = False

    # Only limit the INFERENCE files, not BN-adapt/calibration
    files_for_infer = files[:limit] if (limit and limit > 0) else list(files)

    if patch_size is None:
        patch_size = int(GEN.get("patch_size", 512))
    if stride is None:
        stride = int(GEN.get("stride", 256))

    # ---- Optional BN adaptation (uses full file list) ----
    if bn_adapt_tiles and bn_adapt_tiles > 0:
        n = min(bn_adapt_tiles, len(files))
        print(f"[BN] Adapting BatchNorm on {n} tiles...")
        bn_adapt_update(model, files, mode, device=device, n_tiles=n, scale32768=scale32768)

    # ---- Build debias vector (prefer calibration over zero) ----
    debias_vec = None
    if debias_calib and debias_calib > 0:
        k = min(debias_calib, len(files))
        print(f"[CALIB] Computing debias vector from {k} tiles...")

        # Select evenly spaced tiles across the entire set (deterministic)
        if k == len(files):
            sel_files = list(files)
        else:
            sel_idx = np.linspace(0, len(files) - 1, num=k, dtype=int)
            sel_files = [files[i] for i in sel_idx]

        if debug_fingerprint:
            print("[CALIB] first 5 tiles:", [os.path.basename(p) for p in sel_files[:5]])

        debias_vec = compute_calib_debias_vec(
            model, sel_files, mode, device=device, n_tiles=len(sel_files), scale32768=scale32768
        )

    elif debias_zero:
        if not files:
            raise FileNotFoundError("No files available for zero-input shape inference.")
        if mode in ("tif", "tiff"):
            img, _, _, _ = read_raster(files[0]); H, W = img.shape[1], img.shape[2]
        elif mode == "npy":
            x0, _ = read_npy_pair(files[0]); H, W = x0.shape[1], x0.shape[2]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        x_zero = np.zeros((INPUT_CHANNELS, H, W), dtype=np.float32)
        if scale32768: x_zero = x_zero / 32768.0
        with torch.no_grad():
            xt0 = torch.from_numpy(x_zero).unsqueeze(0).to(device=device, dtype=torch.float32)
            logits0 = model(xt0).squeeze(0)
            debias_vec = logits0.mean(dim=(1, 2)).detach().cpu().numpy()
        if debug_zero:
            probs0 = torch.softmax(logits0, dim=0).cpu().numpy()
            class_means0 = probs0.mean(axis=(1, 2))
            print("[ZERO] per-class softmax mean:", np.round(class_means0, 4).tolist())
            top1 = np.max(probs0, axis=0)
            print("[ZERO] top1 prob mean/min/max:",
                  float(np.mean(top1)), float(np.min(top1)), float(np.max(top1)))

    # ---- Inference loop (on possibly limited subset) ----
    for path in tqdm(files_for_infer, desc="Inference"):
        try:
            # -------- Load inputs --------
            if mode in ("tif", "tiff"):
                img, profile, transform, crs = read_raster(path)
                label_band = None
                if LABEL_IN_FIRST_BAND:
                    if img.shape[0] < (INPUT_CHANNELS + 1):
                        raise ValueError(f"Expected >= {INPUT_CHANNELS+1} bands (label + {INPUT_CHANNELS}), got {img.shape[0]} for {path}")
                    label_band = img[0].copy()
                    x = img[1:1+INPUT_CHANNELS].astype(np.float32, copy=False)
                else:
                    if img.shape[0] < INPUT_CHANNELS:
                        raise ValueError(f"Expected >= {INPUT_CHANNELS} image bands, got {img.shape[0]} for {path}")
                    x = img[:INPUT_CHANNELS].astype(np.float32, copy=False)

                # ---- HOT-FIX for feature nodata ----
                if (x == NODATA_VALUE_INPUT).any():
                    x = x.copy(); x[x == NODATA_VALUE_INPUT] = 0.0
                mask_px_all255 = np.all(x == 255.0, axis=0)
                if mask_px_all255.any():
                    if not x.flags.writeable: x = x.copy()
                    x[:, mask_px_all255] = 0.0

            elif mode == "npy":
                x, label_band = read_npy_pair(path)  # float32 already
                if x.shape[0] != INPUT_CHANNELS:
                    if LABEL_IN_FIRST_BAND and x.shape[0] == INPUT_CHANNELS + 1:
                        label_band = x[0].copy()
                        x = x[1:1+INPUT_CHANNELS]
                    else:
                        x = x[:INPUT_CHANNELS]

                # ---- HOT-FIX for feature nodata ----
                if (x == NODATA_VALUE_INPUT).any():
                    x = x.copy(); x[x == NODATA_VALUE_INPUT] = 0.0
                mask_px_all255 = np.all(x == 255.0, axis=0)
                if mask_px_all255.any():
                    if not x.flags.writeable: x = x.copy()
                    x[:, mask_px_all255] = 0.0
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if scale32768:
                x = x / 32768.0

            # ---- Fingerprint (after both branches set x) ----
            if debug_fingerprint:
                ch0 = x[0]
                sig_mean = float(x.mean())
                sig_std  = float(x.std())
                sig_ch0  = int(zlib.adler32(ch0.tobytes()))
                px0 = np.round(x[:5, 0, 0], 1).tolist()
                print(f"[SIG] {os.path.basename(path)} mean={sig_mean:.2f} std={sig_std:.2f} "
                      f"adler32(ch0)={sig_ch0} px0[:5]={px0}")

            # ---- One-time input stats ----
            if not printed_stats:
                x_min, x_max = float(np.min(x)), float(np.max(x))
                x_mean, x_std = float(np.mean(x)), float(np.std(x))
                frac_zero = float(np.mean(np.all(x == 0.0, axis=0)))
                print(f"[STATS] {os.path.basename(path)} → min={x_min:.2f}, max={x_max:.2f}, "
                      f"mean={x_mean:.2f}, std={x_std:.2f}, allzero_px={frac_zero*100:.2f}%")
                printed_stats = True

            # -------- Predict (sliding vs full-tile) --------
            if sliding:
                pred, probs = predict_tile_sliding(
                    model, x, patch_size=patch_size, stride=stride, device=device,
                    debias_vec=debias_vec, temperature=temperature
                )
            else:
                pred, probs = predict_tile(
                    model, x, device=device,
                    debias_vec=debias_vec, temperature=temperature
                )

            # ---- Prob diagnostics ----
            class_means = probs.mean(axis=(1, 2))
            top1 = np.max(probs, axis=0)
            print("[PROBMEAN]", os.path.basename(path), "→", np.round(class_means, 4).tolist(),
                  "| top1 mean/min/max:",
                  float(np.mean(top1)), float(np.min(top1)), float(np.max(top1)))

            # -------- Propagate ignore (255) from label if available --------
            if label_band is not None:
                mask_ignore = (label_band == NODATA_LABEL)
                if mask_ignore.any():
                    pred = pred.copy(); pred[mask_ignore] = NODATA_LABEL

            # -------- Save outputs --------
            if mode in ("tif", "tiff"):
                out_path = os.path.join(out_dir, os.path.basename(path).replace(".tif", "_pred.tif").replace(".tiff",
                                                                                                             "_pred.tif"))
                write_raster(out_path, pred, profile, transform)
            else:
                out_path = write_npy_pred(path, pred, out_dir)

            if SAVE_PROBS:
                np.savez_compressed(out_path.replace("_pred.tif", "_probs.npz").replace("_pred.npy", "_probs.npz"),
                                    probs=probs.astype(np.float32))

            # -------- Quick stats --------
            valid = pred[pred != NODATA_LABEL]
            counts = collections.Counter(valid.flatten().tolist())
            print(f"[DEBUG] {os.path.basename(path)} → valid pix: {valid.size}, class hist: {dict(counts)}")

        except Exception as e:
            print(f"[ERROR] {path}: {e}")


# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True, help="Run timestamp (e.g. 20250806_153626)")
    parser.add_argument("--input_glob", default=None, help="Optional glob pattern to override input listing")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files (debug)")
    parser.add_argument("--scale32768", action="store_true",
                        help="Divide inputs by 32768.0 before inference (quick normalization test)")
    parser.add_argument("--debug_zero", action="store_true",
                        help="Run a single forward on an all-zero tile to inspect classifier bias.")
    parser.add_argument("--debias_zero", action="store_true",
                        help="Subtract per-class logits from a zero-input forward before softmax.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature scaling factor applied to logits (after optional debias).")
    parser.add_argument("--bn_adapt_tiles", type=int, default=0,
                        help="Update BatchNorm stats using N tiles before inference (no grads).")
    parser.add_argument("--debias_calib", type=int, default=0,
                        help="Compute debias vector from average logits over N tiles (uses real data).")
    parser.add_argument("--debug_fingerprint", action="store_true",
                        help="Print per-tile input fingerprints to verify tiles differ.")
    parser.add_argument("--sliding", action="store_true",
                        help="Use sliding-window inference with overlap (legacy behavior).")
    parser.add_argument("--patch_size", type=int, default=GEN.get("patch_size", 512),
                        help="Sliding window patch size.")
    parser.add_argument("--stride", type=int, default=GEN.get("stride", 256),
                        help="Sliding window stride.")

    args = parser.parse_args()

    run_dir = resolve_outputs_dir(args.timestamp)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    weights_path = find_best_checkpoint(run_dir)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    print(f"[INFO] Using weights: {os.path.basename(weights_path)}")
    model = load_model(weights_path, device=args.device)

    # ---------- Assemble files ----------
    if args.input_glob:
        files = sorted(glob.glob(args.input_glob, recursive=True))
        mode = None
        if files:
            ext = os.path.splitext(files[0])[1].lower()
            mode = "tif" if ext in (".tif", ".tiff") else "npy" if ext == ".npy" else None
    else:
        input_root = resolve_input_root(args.timestamp)
        files, mode = discover_files(input_root)

    if not files or mode is None:
        raise FileNotFoundError("No input files found (tried *.tif, *.tiff, *_img.npy).")

    # Avoid re-picking up our own outputs
    files = [f for f in files if "_pred.tif" not in f and "_probs.npz" not in f]
    pred_dir = os.path.join(run_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    run_inference_on_files(model, files, mode, pred_dir,
                           device=args.device,
                           limit=args.limit,
                           scale32768=args.scale32768,
                           debug_zero=args.debug_zero,
                           debias_zero=args.debias_zero,
                           debias_calib=args.debias_calib,
                           bn_adapt_tiles=args.bn_adapt_tiles,
                           temperature=args.temperature,
                           debug_fingerprint=args.debug_fingerprint,
                           sliding=args.sliding,
                           patch_size=args.patch_size,
                           stride=args.stride)


if __name__ == "__main__":
    main()
