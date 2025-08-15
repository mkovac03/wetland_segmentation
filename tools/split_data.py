# -*- coding: utf-8 -*-
"""
tools/split_data.py

Augment a VERIFIED selection with:
  A) class stats + weights (from labels/label_histograms.jsonl, only for selected tiles)
  B) UTM-stratified train/val/test splits with per-zone representation guarantees
  C) per-channel mean/std from TRAIN inputs (TIFF/VRT or NPY), strictly nodata-masked

Writes BACK into the same verified splits JSON and saves two figures.

Usage:
  python tools/split_data.py \
    --config configs/config.yaml \
    --verified data/splits/splits_verified_YYYYMMDD_HHMMSS.json \
    --weights-method median_frequency \
    --beta 0.9999 \
    --exts tif,tiff,vrt,npy \
    --plots-out outputs/split_stats_YYYYMMDD_HHMMSS \
    [--assert-zone-coverage]
"""

import os, re, json, argparse, yaml, math, random
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm
import rasterio

# Use non-interactive backend (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Key & UTM helpers --------------------
def extract_key_from_path(path: str):
    """
    Your naming patterns:
      google_embed_..._32638_bands_00_63_99996.tif  -> 32638_99996
      ext_wetland_..._32638_99996.tif               -> 32638_99996
    Logic:
      - zone  = last token matching 32[67]\\d{2}
      - tile  = last numeric token in stem (if equals zone, use previous numeric)
    """
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    nums = re.findall(r'\d+', stem)
    if not nums:
        return None
    zones = [n for n in nums if re.fullmatch(r'32[67]\d{2}', n)]
    if not zones:
        return None
    zone = zones[-1]
    tail_ids = re.findall(r'_(\d+)', stem)
    tile = tail_ids[-1] if tail_ids else nums[-1]
    if tile == zone and len(nums) >= 2:
        tile = nums[-2]
    if tile == zone:
        return None
    return f"{zone}_{tile}"

def utm_zone_from_key(key: str):
    p = key.split("_", 1)
    return p[0] if len(p) == 2 else "unknown"

# -------------------- Config / paths --------------------
def load_cfg(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

def resolve_base_paths(cfg):
    base = cfg.get("input_dir")
    if not base:
        raise SystemExit("[ERR] config.input_dir is not set.")
    inputs_root = os.path.join(base, "inputs")
    labels_root = os.path.join(base, "labels")
    jsonl_path = os.path.join(labels_root, "label_histograms.jsonl")
    return inputs_root, labels_root, jsonl_path

# -------------------- Index inputs by key --------------------
def index_inputs(inputs_root, exts):
    idx = {}
    if not (inputs_root and os.path.isdir(inputs_root)):
        return idx
    exts = tuple(e.lower() for e in exts)
    for dp, _, files in os.walk(inputs_root):
        for f in files:
            if f.lower().endswith(exts):
                full = os.path.join(dp, f)
                k = extract_key_from_path(full)
                if k:
                    idx[k] = full
    return idx

# -------------------- Histograms JSONL --------------------
def load_hist_jsonl(jsonl_path):
    """Return dict: key -> {'hist': {int: int}, 'total': int}"""
    out = {}
    if not os.path.exists(jsonl_path):
        raise SystemExit(f"[ERR] Missing histogram JSONL: {jsonl_path}")
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            k = rec.get("tile_key")
            if not k:
                for fld in ("label_path", "path", "file"):
                    if fld in rec:
                        k = extract_key_from_path(rec[fld])
                        if k: break
            if not k:
                continue
            hist = rec.get("hist") or rec.get("histogram") or {}
            h2, tot = {}, 0
            if isinstance(hist, dict):
                for a, b in hist.items():
                    try:
                        ai = int(a); bi = int(b)
                    except Exception:
                        continue
                    h2[ai] = h2.get(ai, 0) + bi
                    tot += bi
            if "total" in rec:
                try: tot = int(rec["total"])
                except Exception: pass
            out[k] = {"hist": h2, "total": int(tot)}
    return out

def aggregate_class_counts(selected_keys, hist_cache, num_classes, ignore_val):
    counts = np.zeros(num_classes, dtype=np.int64)
    covered = 0
    missing = 0
    for k in selected_keys:
        rec = hist_cache.get(k)
        if not rec:
            missing += 1
            continue
        covered += 1
        for cls, cnt in rec["hist"].items():
            if cls == ignore_val:
                continue  # drop ignore pixels
            if 0 <= cls < num_classes:
                counts[cls] += int(cnt)
            # silently drop out-of-range codes
    return counts, covered, missing

# -------------------- Class weights --------------------
def compute_class_weights(counts: np.ndarray, method: str, beta: float = 0.9999):
    total = counts.sum()
    eps = 1e-12
    if method == "inverse_frequency":
        C = max(1, len(counts))
        w = (total / (counts + eps)) / C
        w = w / (w.mean() + eps)
        meta = {"method": method}
    elif method == "median_frequency":
        freq = counts.astype(np.float64) / (total + eps)
        nz = freq[freq > 0]
        if len(nz) == 0:
            w = np.ones_like(freq)
        else:
            med = np.median(nz)
            w = np.where(freq > 0, med / (freq + eps), 0.0)
            w = w / (w.mean() + eps)
        meta = {"method": method}
    elif method == "effective_number":
        eff = (1.0 - beta) / (1.0 - np.power(beta, counts.astype(np.float64) + eps))
        w = 1.0 / (eff + eps)
        w = w / (w.mean() + eps)
        meta = {"method": method, "beta": beta}
    else:
        raise ValueError(f"Unknown weights-method: {method}")
    return w.tolist(), meta

# -------------------- UTM-aware splitting with guarantees --------------------
def _allocate_zone_counts(n, r_train, r_val, r_test):
    """
    Allocate n items across (train, val, test) with:
      - proportional to ratios (r_*),
      - largest remainder rounding,
      - ensure at least 1 item in each split whose ratio > 0 whenever feasible
        (i.e., if n >= number_of_positive_ratios).
    Returns (n_tr, n_va, n_te) summing to n.
    """
    import math
    splits = [("train", r_train), ("val", r_val), ("test", r_test)]
    pos = [(name, r) for name, r in splits if r > 0]
    if not pos:
        return n, 0, 0  # degenerate: all to train

    total_r = sum(r for _, r in pos)
    raw = {name: (n * r / total_r) for name, r in pos}
    floors = {name: int(math.floor(raw[name])) for name, _ in pos}
    fracs  = {name: (raw[name] - floors[name]) for name, _ in pos}
    alloc  = floors.copy()
    remain = n - sum(alloc.values())

    # Ensure ≥1 per positive-ratio split if feasible
    zeros = [name for name, _ in pos if alloc[name] == 0]
    need = len(zeros)
    if need > 0 and n >= len(pos):
        zeros_sorted = sorted(zeros, key=lambda nm: dict(pos)[nm], reverse=True)
        give = min(remain, need)
        for nm in zeros_sorted[:give]:
            alloc[nm] += 1
            remain -= 1

    # Distribute the rest by largest fractional remainder
    if remain > 0:
        order = sorted(pos, key=lambda t: fracs[t[0]], reverse=True)
        i = 0
        while remain > 0:
            alloc[order[i % len(order)][0]] += 1
            remain -= 1
            i += 1

    return alloc.get("train", 0), alloc.get("val", 0), alloc.get("test", 0)

def _counts_by_utm(keys):
    c = defaultdict(int)
    for k in keys:
        c[utm_zone_from_key(k)] += 1
    # sort by count desc then zone
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))

def split_stratified_by_utm(keys, r_train, r_val, r_test, seed, strategy, max_train=None):
    """
    Stratify by UTM zone with per-zone proportional allocation and guarantees:
      - For each zone (size n), allocate (n_tr, n_va, n_te) using _allocate_zone_counts.
      - If utm_sampling_strategy == "equal" with max_training_tiles:
          * keep ≥1 train tile per zone if possible
          * if max_training_tiles < #zones, warn and keep one per zone until cap
    """
    rnd = random.Random(seed)
    zones = defaultdict(list)
    for k in keys:
        zones[utm_zone_from_key(k)].append(k)

    train, val, test = [], [], []
    denom = (r_train + r_val + r_test)

    for z, ks in zones.items():
        ks = list(ks); rnd.shuffle(ks)
        n = len(ks)
        n_tr, n_va, n_te = _allocate_zone_counts(n, r_train, r_val, r_test)
        t0, t1 = 0, n_tr
        v0, v1 = t1, t1 + n_va
        te0, te1 = v1, v1 + n_te
        train.extend(ks[t0:t1])
        val.extend(ks[v0:v1])
        test.extend(ks[te0:te1])

    # Optional equalization + cap for TRAIN
    if strategy == "equal" and max_train is not None:
        # bucket current train by zone
        buckets = {}
        for z in zones:
            buckets[z] = [k for k in train if utm_zone_from_key(k) == z]
            rnd.shuffle(buckets[z])

        Z = len(zones)
        if max_train < Z:
            print(f"[WARN] max_training_tiles({max_train}) < #zones({Z}); cannot keep 1 train tile per zone.")
            new_train = []
            for z in sorted(buckets.keys()):
                if buckets[z]:
                    new_train.append(buckets[z][0])
                    if len(new_train) >= max_train:
                        break
            train = new_train
        else:
            seed_set = [b[0] for b in buckets.values() if b]
            remaining = max_train - len(seed_set)
            leftovers = [b[1:] for b in buckets.values()]
            new_train = list(seed_set)
            while remaining > 0 and any(leftovers):
                for i in range(len(leftovers)):
                    if leftovers[i]:
                        new_train.append(leftovers[i].pop(0))
                        remaining -= 1
                        if remaining == 0:
                            break
            train = new_train

    # ensure uniqueness (no-op if keys unique)
    train = list(dict.fromkeys(train))
    val   = list(dict.fromkeys(val))
    test  = list(dict.fromkeys(test))

    return train, val, test

def _probe_channels(path, nodata_cfg):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff", ".vrt"):
        with rasterio.open(path) as src:
            return src.count
    elif ext == ".npy":
        arr = np.load(path, mmap_mode="r")
        return int(arr.shape[0])
    else:
        raise ValueError(f"Unsupported input extension: {ext}")

def _stats_one_file(args):
    """Worker: return (sums[C], sumsqs[C], counts[C]) for a single file."""
    fp, C, nodata_cfg = args
    ext = os.path.splitext(fp)[1].lower()
    sums = np.zeros(C, dtype=np.float64)
    sumsqs = np.zeros(C, dtype=np.float64)
    counts = np.zeros(C, dtype=np.int64)

    if ext in (".tif", ".tiff", ".vrt"):
        with rasterio.open(fp) as src:
            nod = src.nodata if src.nodata is not None else nodata_cfg
            for i in range(C):
                band = src.read(i+1).astype(np.float32)
                valid = np.isfinite(band)
                if nod is not None:
                    valid &= (band != nod)
                if not np.any(valid):
                    continue
                x = band[valid].astype(np.float64)
                sums[i]   += float(x.sum())
                sumsqs[i] += float((x * x).sum())
                counts[i] += int(valid.sum())
    elif ext == ".npy":
        arr = np.load(fp, mmap_mode="r")  # [C,H,W]
        if arr.ndim != 3 or arr.shape[0] < C:
            return sums, sumsqs, counts
        for i in range(C):
            ch = arr[i].astype(np.float32)
            valid = np.isfinite(ch)
            if nodata_cfg is not None:
                valid &= (ch != nodata_cfg)
            if not np.any(valid):
                continue
            x = ch[valid].astype(np.float64)
            sums[i]   += float(x.sum())
            sumsqs[i] += float((x * x).sum())
            counts[i] += int(valid.sum())
    else:
        # unsupported; return zeros
        pass
    # Return small Python lists (pickle-friendly)
    return sums.tolist(), sumsqs.tolist(), counts.tolist()


# -------------------- Channel stats (strict nodata mask) --------------------
def compute_channel_stats(train_keys, inputs_index, exts, nodata_cfg, expected_channels=None, workers=0):
    """Parallel channel stats over TRAIN.
    workers: 0/1 -> single process; -1 -> all cores; N -> exactly N processes.
    """
    files = [inputs_index[k] for k in train_keys if k in inputs_index]
    if not files:
        return {"channels": 0, "mean": [], "std": [], "n_valid": []}

    # Decide channel count once
    C = expected_channels or _probe_channels(files[0], nodata_cfg)

    # Accumulators
    sums   = np.zeros(C, dtype=np.float64)
    sumsqs = np.zeros(C, dtype=np.float64)
    counts = np.zeros(C, dtype=np.int64)

    if workers in (0, 1):
        # single-process (original behavior)
        for fp in tqdm(files, desc="Channel stats (TRAIN)"):
            s, sq, ct = _stats_one_file((fp, C, nodata_cfg))
            sums   += np.asarray(s,  dtype=np.float64)
            sumsqs += np.asarray(sq, dtype=np.float64)
            counts += np.asarray(ct, dtype=np.int64)
    else:
        nproc = cpu_count() if workers == -1 else max(1, int(workers))
        # Small chunk to keep workers busy but reduce overhead
        chunk = 4
        with Pool(processes=nproc) as pool:
            for s, sq, ct in tqdm(
                pool.imap_unordered(_stats_one_file, ((fp, C, nodata_cfg) for fp in files), chunksize=chunk),
                total=len(files),
                desc=f"Channel stats (TRAIN) x{nproc}"
            ):
                sums   += np.asarray(s,  dtype=np.float64)
                sumsqs += np.asarray(sq, dtype=np.float64)
                counts += np.asarray(ct, dtype=np.int64)

    means = np.zeros(C, dtype=np.float64)
    stds  = np.zeros(C, dtype=np.float64)
    for i in range(C):
        if counts[i] > 0:
            mu  = sums[i] / counts[i]
            var = max(0.0, (sumsqs[i] / counts[i]) - mu * mu)
            means[i] = mu
            stds[i]  = math.sqrt(var)
        else:
            means[i], stds[i] = 0.0, 1.0  # safe defaults
    return {
        "channels": int(C),
        "mean":  [float(x) for x in means],
        "std":   [float(x) for x in stds],
        "n_valid": [int(x) for x in counts],
        "workers": workers,
    }


# -------------------- Plotting --------------------
def plot_class_weights(counts, weights, label_names, out_png):
    classes = list(range(len(counts)))
    names = [label_names.get(i, str(i)) for i in classes]
    fig, ax1 = plt.subplots(figsize=(max(8, len(classes)*0.6), 4), dpi=150)
    ax1.bar(classes, [weights[i] for i in classes], alpha=0.9)
    ax1.set_title("Class Weights")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Weight")
    ax1.set_xticks(classes)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def plot_channel_stats(means, stds, out_png):
    x = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(max(8, len(x)*0.2), 4), dpi=150)
    ax.plot(x, means, marker=".", linewidth=1, label="mean")
    ax.plot(x, stds,  marker=".", linewidth=1, label="std")
    ax.set_title("Channel-wise Mean / Std (TRAIN)")
    ax.set_xlabel("Channel index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

# ---- Keep ratios when max_training_tiles caps train (downscale val/test) ----
def _downsample_preserving_zones(keys, target, seed):
    """Round-robin sample across UTM zones to hit 'target' while keeping ≥1 per zone when feasible."""
    from collections import defaultdict
    rnd = random.Random(seed)
    byz = defaultdict(list)
    for k in keys: byz[utm_zone_from_key(k)].append(k)
    # shuffle within zones
    for z in byz: rnd.shuffle(byz[z])
    zones = list(byz.keys())

    # If we can keep ≥1 per zone, seed with one each
    picked = []
    if target >= len(zones):
        for z in zones:
            if byz[z]:
                picked.append(byz[z].pop(0))
    # fill remainder round-robin
    while len(picked) < target and any(byz.values()):
        for z in zones:
            if byz[z]:
                picked.append(byz[z].pop(0))
                if len(picked) >= target:
                    break
    return picked

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--verified", required=True, help="verified selection JSON to augment (in-place update)")
    ap.add_argument("--weights-method", default="median_frequency",
                    choices=["median_frequency", "inverse_frequency", "effective_number"])
    ap.add_argument("--beta", type=float, default=0.9999, help="beta for effective_number")
    ap.add_argument("--exts", default="tif,tiff,vrt,npy", help="input file extensions under <input_dir>/inputs")
    ap.add_argument("--plots-out", default=None, help="directory to save plots; default uses timestamp next to verified file")
    ap.add_argument("--assert-zone-coverage", action="store_true",
                    help="fail if any UTM zone is missing from any split")
    ap.add_argument("--workers", type=int, default=0,
                    help="workers for channel stats (0/1=single process, -1=all cores, N=that many)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    inputs_root, labels_root, jsonl_path = resolve_base_paths(cfg)
    num_classes = int(cfg.get("num_classes", 13))
    ignore_val  = int(cfg.get("ignore_val", 255))
    nodata_cfg  = cfg.get("nodata_val", None)
    expected_C  = int(cfg.get("input_channels", 64))
    label_names_cfg = cfg.get("label_names", {}) or {}
    label_names = {int(k): v for k, v in label_names_cfg.items()}

    # Load verified selection
    with open(args.verified, "r") as f:
        meta = json.load(f)
    selected = list(meta.get("selected_tile_keys", []))
    if not selected:
        raise SystemExit("[ERR] verified splits has no 'selected_tile_keys'.")

    # ---- A) class stats + weights
    print("[A] Aggregating class counts from label_histograms.jsonl (selected tiles only)…")
    hist_cache = load_hist_jsonl(jsonl_path)
    counts, covered, missing = aggregate_class_counts(selected, hist_cache, num_classes, ignore_val)
    print(f"    Covered tiles in JSONL: {covered} / {len(selected)}  (missing: {missing})")
    weights_list, wm = compute_class_weights(counts, args.weights_method, beta=args.beta)
    weights = {i: float(weights_list[i]) for i in range(num_classes)}

    meta["class_stats"] = {
        "num_classes": num_classes,
        "ignore_val": ignore_val,
        "counts": {str(i): int(c) for i, c in enumerate(counts.tolist())},
        "total_pixels": int(counts.sum()),
        "weights": {**wm, "values": {str(i): weights[i] for i in range(num_classes)}}
    }

    # ---- B) stratified train/val/test with per-zone guarantees
    print("[B] Creating stratified splits (UTM-aware with per-zone representation)…")
    sp = cfg.get("splitting", {}) or {}
    r_train = float(sp.get("train_ratio", 1))
    r_val   = float(sp.get("val_ratio", 0.1))
    r_test  = float(sp.get("test_ratio", 0.1))
    seed    = int(sp.get("seed", 42))
    strat   = bool(sp.get("stratify_by_utm", True))
    strategy= sp.get("utm_sampling_strategy", "equal")
    max_tr  = sp.get("max_training_tiles", None)

    if strat:
        train, val, test = split_stratified_by_utm(selected, r_train, r_val, r_test, seed,
                                                   strategy=strategy, max_train=max_tr)
    else:
        rnd = random.Random(seed)
        xs = list(selected); rnd.shuffle(xs)
        denom = (r_train + r_val + r_test)
        n = len(xs)
        n_tr = int(round(n * r_train / denom))
        n_va = int(round(n * r_val   / denom))
        train = xs[:n_tr]; val = xs[n_tr:n_tr+n_va]; test = xs[n_tr+n_va:]

    meta["train"] = train
    meta["val"]   = val
    meta["test"]  = test

    if cfg.get("splitting", {}).get("max_training_tiles") is not None:
        r_train = float(sp.get("train_ratio", 1))
        r_val = float(sp.get("val_ratio", 0.1))
        r_test = float(sp.get("test_ratio", 0.1))
        # target sizes to preserve r_val/r_train and r_test/r_train relative to *capped* train
        tgt_val = int(round(len(train) * (r_val / r_train))) if r_train > 0 else len(val)
        tgt_test = int(round(len(train) * (r_test / r_train))) if r_train > 0 else len(test)

        # downsample val/test to targets while preserving UTM coverage
        if len(val) > tgt_val:
            val = _downsample_preserving_zones(val, tgt_val, seed)
        if len(test) > tgt_test:
            test = _downsample_preserving_zones(test, tgt_test, seed)

        meta["val"] = val
        meta["test"] = test

    # Add UTM counts to JSON for sanity checks
    train_counts = _counts_by_utm(train)
    val_counts   = _counts_by_utm(val)
    test_counts  = _counts_by_utm(test)
    meta["train_utm_counts"] = train_counts
    meta["val_utm_counts"]   = val_counts
    meta["test_utm_counts"]  = test_counts

    print(f"    train={len(train)}  val={len(val)}  test={len(test)}  total={len(train)+len(val)+len(test)}")
    print("[DIAG] UTM zones per split:",
          f"train={len(train_counts)}  val={len(val_counts)}  test={len(test_counts)}")

    if args.assert_zone_coverage:
        zones_all = set(_counts_by_utm(selected).keys())
        for name, counts_dict in [("train", train_counts), ("val", val_counts), ("test", test_counts)]:
            missing_zones = zones_all - set([z for z, c in counts_dict.items() if c > 0])
            if missing_zones:
                raise SystemExit(f"[ASSERT] Missing zones in {name}: {sorted(missing_zones)}")

    # ---- C) channel-wise mean/std on TRAIN (strict nodata mask)
    print("[C] Computing channel-wise mean/std on TRAIN (nodata-masked)…")
    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    inputs_index = index_inputs(inputs_root, exts)
    if not inputs_index:
        raise SystemExit(f"[ERR] No inputs indexed under: {inputs_root}. Check files & extensions.")

    missing_inputs = [k for k in train if k not in inputs_index]
    if missing_inputs:
        print(f"[WARN] {len(missing_inputs)} train tiles missing in inputs (first 10): {missing_inputs[:10]}")

    norm = compute_channel_stats(train, inputs_index, exts, nodata_cfg,
                                 expected_channels=expected_C, workers=args.workers)
    meta["normalization"] = norm

    # ---- Plots
    token = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_plot_dir = os.path.join("outputs", f"split_stats_{token}")
    plot_dir = args.plots_out or default_plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    cw_png  = os.path.join(plot_dir, "class_weights.png")
    ch_png  = os.path.join(plot_dir, "channel_mean_std.png")

    print("[PLOTS] Writing figures…")
    plot_class_weights(counts, weights, {int(k): v for k, v in label_names.items()}, cw_png)
    plot_channel_stats(norm.get("mean", []), norm.get("std", []), ch_png)

    # ---- Write back (with one-time .bak) + figure paths
    meta.setdefault("figures", {})
    meta["figures"]["class_weights_png"] = cw_png
    meta["figures"]["channel_mean_std_png"] = ch_png

    bak = args.verified + ".bak"
    if not os.path.exists(bak):
        try:
            import shutil; shutil.copyfile(args.verified, bak)
            print(f"[OK] Backup saved → {bak}")
        except Exception:
            pass

    with open(args.verified, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Updated verified splits → {args.verified}")
    print(f"[OK] Saved figures → {plot_dir}")
    print("Done.")

if __name__ == "__main__":
    main()
