# -*- coding: utf-8 -*-
# Strict visualizer with reprojection to EPSG:3035, robust nodata masking/stretch,
# and tight cropping (no black/white margins).
#
# Run:
#   python tools/viz_random_tiles.py \
#     --config configs/config.yaml \
#     --splits data/splits/splits_verified_YYYYMMDD_HHMMSS.json \
#     --set selected --num 16 --grid

import os, re, json, argparse, yaml, random
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

RX_NOW = re.compile(r'(\d{8}_\d{6})')

def extract_key_from_path(path: str):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
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

def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def index_by_key(root, exts):
    idx = {}
    if not root or not os.path.isdir(root):
        return idx
    exts = tuple(e.lower() for e in exts)
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                full = os.path.join(dp, f)
                k = extract_key_from_path(full)
                if k: idx[k] = full
    return idx

def build_label_cmap(cfg):
    names = cfg.get("label_names", {})
    base = plt.get_cmap("tab20").colors
    max_id = max((int(k) for k in names.keys()), default=19)
    colors = [base[i % len(base)] for i in range(max_id + 1)]
    return ListedColormap(colors)

def robust_stretch(channel: np.ndarray, mask: np.ndarray, pmin: float, pmax: float):
    ch = channel.astype(np.float32, copy=False)
    valid = ~mask & np.isfinite(ch)
    if valid.sum() == 0:
        out = np.zeros_like(ch, dtype=np.float32); return out
    lo = np.percentile(ch[valid], pmin)
    hi = np.percentile(ch[valid], pmax)
    if not np.isfinite(lo): lo = np.nanmin(ch[valid])
    if not np.isfinite(hi): hi = np.nanmax(ch[valid])
    if hi <= lo:
        out = np.zeros_like(ch, dtype=np.float32)
        out[valid] = 1.0
        return out
    out = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    out[~valid] = 0.0
    return out.astype(np.float32)

def reproject_label_to_3035(label_path, dst_crs, dst_res, ignore_val):
    with rasterio.open(label_path) as src:
        src_crs = src.crs
        src_bounds = src.bounds
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height,
            left=src_bounds.left, bottom=src_bounds.bottom,
            right=src_bounds.right, top=src_bounds.top,
            resolution=dst_res
        )
        dst = np.full((dst_h, dst_w), ignore_val, dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src_crs,
            src_nodata=src.nodata if src.nodata is not None else ignore_val,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=ignore_val,
            resampling=Resampling.nearest
        )
    return dst, dst_transform

def reproject_rgb_to_grid(rgb_path, dst_crs, dst_transform, dst_shape, nodata_cfg, pmin, pmax):
    H, W = dst_shape
    ext = os.path.splitext(rgb_path)[1].lower()
    if ext not in (".tif", ".tiff", ".vrt"):
        raise SystemExit("[ERR] Reprojection needs georeferenced inputs (TIFF/VRT).")
    with rasterio.open(rgb_path) as src:
        nb = min(3, src.count)
        dst_stack = np.full((nb, H, W), np.nan, dtype=np.float32)
        for i in range(nb):
            reproject(
                source=rasterio.band(src, i+1),
                destination=dst_stack[i],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata if src.nodata is not None else nodata_cfg,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear
            )
    # nodata mask from NaNs and configured nodata
    mask = np.isnan(dst_stack[0])
    for i in range(1, nb):
        mask |= np.isnan(dst_stack[i])
    if nodata_cfg is not None:
        for i in range(nb):
            mask |= (dst_stack[i] == nodata_cfg)
    # stretch
    rgb = np.zeros((3, H, W), dtype=np.float32)
    for i in range(nb):
        rgb[i] = robust_stretch(dst_stack[i], mask, pmin, pmax)
    if nb < 3:
        for i in range(nb, 3):
            rgb[i] = rgb[nb-1]
    rgb_img = np.transpose(rgb[:3], (1, 2, 0))  # HWC
    return rgb_img, mask

def tight_crop(rgb_img: np.ndarray, rgb_mask: np.ndarray, lab: np.ndarray, ignore_val: int):
    """Crop both arrays to the minimal bbox that contains ANY valid pixels from RGB or labels."""
    valid_rgb = ~rgb_mask
    valid_lab = (lab != ignore_val)
    valid_any = valid_rgb | valid_lab
    if not np.any(valid_any):
        return rgb_img, lab  # nothing valid, skip crop
    rows = np.where(valid_any.any(axis=1))[0]
    cols = np.where(valid_any.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    return rgb_img[r0:r1, c0:c1, :], lab[r0:r1, c0:c1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True, help="verified selection JSON or final splits JSON")
    ap.add_argument("--set", default="selected", choices=["selected","train","val","test"])
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--exts", default="tif,tiff,vrt")
    ap.add_argument("--pmin", type=float, default=2.0)
    ap.add_argument("--pmax", type=float, default=98.0)
    args = ap.parse_args()

    random.seed(args.seed)
    cfg = load_cfg(args.config)
    base = cfg.get("input_dir")
    if not base:
        raise SystemExit("[ERR] config.input_dir is not set.")
    inputs_root = os.path.join(base, "inputs")
    labels_root = os.path.join(base, "labels")
    if not os.path.isdir(inputs_root) or not os.path.isdir(labels_root):
        raise SystemExit(f"[ERR] Missing inputs/labels under base: {base}")

    dst_crs = cfg.get("crs_target", "EPSG:3035")
    dst_res = 10
    nodata_cfg = cfg.get("nodata_val", None)   # e.g., -32768
    ignore_val = cfg.get("ignore_val", 255)

    with open(args.splits, "r") as f:
        data = json.load(f)
    keys = (list(data.get("selected_tile_keys", [])) if args.set == "selected"
            else list(data.get(args.set, [])))
    if not keys:
        raise SystemExit(f"[ERR] No keys found for set '{args.set}' in {args.splits}")

    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    input_idx = index_by_key(inputs_root, exts)
    label_idx = index_by_key(labels_root, ("tif","tiff","vrt"))

    both = [k for k in keys if (k in input_idx and k in label_idx)]
    if not both:
        print(f"[DIAG] Example selected (first 10): {keys[:10]}")
        print(f"[DIAG] Example inputs (first 10): {list(input_idx.keys())[:10]}")
        print(f"[DIAG] Example labels (first 10): {list(label_idx.keys())[:10]}")
        raise SystemExit("[ERR] No overlapping tiles between inputs and labels (strict mode).")

    token_match = RX_NOW.search(os.path.basename(args.splits))
    token = token_match.group(1) if token_match else "now"
    outdir = args.outdir or f"outputs/quickviz_{token}"
    os.makedirs(outdir, exist_ok=True)

    cmap = build_label_cmap(cfg)
    sample = random.sample(both, min(args.num, len(both)))
    print(f"[INFO] Sampling {len(sample)} tiles (from {len(both)}). Reprojecting to {dst_crs} @ {dst_res}m. Saving → {outdir}")

    grid_paths = []
    for k in sample:
        p_in, p_lb = input_idx[k], label_idx[k]

        # label grid in 3035
        lab_3035, dst_transform = reproject_label_to_3035(p_lb, dst_crs, dst_res, ignore_val)
        H, W = lab_3035.shape

        # rgb to same grid
        rgb_img, rgb_mask = reproject_rgb_to_grid(p_in, dst_crs, dst_transform, (H, W), nodata_cfg, args.pmin, args.pmax)

        # tight crop to valid area
        rgb_img_c, lab_c = tight_crop(rgb_img, rgb_mask, lab_3035, ignore_val)
        lab_vis = np.ma.masked_where(lab_c == ignore_val, lab_c)

        # draw without any padding
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
        for ax in axs:
            ax.set_axis_off()
            ax.margins(0)
        axs[0].imshow(rgb_img_c, interpolation="nearest")
        axs[1].imshow(lab_vis, cmap=cmap, interpolation="nearest")

        out_png = os.path.join(outdir, f"{k}.png")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        grid_paths.append(out_png)

    if args.grid and grid_paths:
        cols = 4
        rows = int(np.ceil(len(grid_paths) / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3), dpi=150)
        axs = np.atleast_2d(axs)
        for i, ax in enumerate(axs.flatten()):
            ax.set_axis_off(); ax.margins(0)
            if i < len(grid_paths):
                img = plt.imread(grid_paths[i]); ax.imshow(img)
        grid_out = os.path.join(outdir, f"grid_{args.set}.png")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
        fig.savefig(grid_out, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"[OK] Wrote grid → {grid_out}")

    print(f"[OK] Wrote {len(grid_paths)} previews → {outdir}")

if __name__ == "__main__":
    main()
