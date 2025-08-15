# File: split_data.py
import os
import re
import json
import yaml
import numpy as np
import hashlib
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# -------------------------
# Helpers
# -------------------------
ZONE_TILE_RE = re.compile(r"_(\d{5})_(\d+)(?:\.tif|\.tiff)$", re.IGNORECASE)

def parse_zone_tile(path: str):
    m = ZONE_TILE_RE.search(path)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def get_split_hash(cfg):
    split_cfg = cfg.get("splitting", {})
    hash_input = json.dumps({
        "train_ratio": split_cfg.get("train_ratio", 0.8),
        "val_ratio":   split_cfg.get("val_ratio",   0.1),
        "test_ratio":  split_cfg.get("test_ratio",  0.1),
        "seed":        split_cfg.get("seed", 42),
        "num_classes": cfg.get("num_classes", 13),
        "stratify_by_utm": split_cfg.get("stratify_by_utm", False),
        "utm_sampling_strategy": split_cfg.get("utm_sampling_strategy", "proportional"),
    }, sort_keys=True)
    return hashlib.md5(hash_input.encode()).hexdigest()[:8]

def resolve_now_from_config(cfg: dict):
    proc = cfg.get("processed_dir", "data/processed/{now}")
    if "{now}" in proc:
        base = proc.split("{now}")[0].rstrip("/ ")
        now = None
        if os.path.isdir(base):
            cands = [d for d in os.listdir(base)
                     if re.match(r"^\d{8}_\d{6}$", d) and os.path.isdir(os.path.join(base, d))]
            now = sorted(cands)[-1] if cands else None
        if now is None:
            leaf = os.path.basename(os.path.normpath(proc))
            now = leaf if re.match(r"^\d{8}_\d{6}$", leaf) else "00000000_000000"
        processed_dir = proc.replace("{now}", now)
    else:
        processed_dir = proc
        leaf = os.path.basename(os.path.normpath(processed_dir))
        now = leaf if re.match(r"^\d{8}_\d{6}$", leaf) else "00000000_000000"

    out_tpl = cfg.get("output_dir", "outputs/{now}")
    output_dir = out_tpl.replace("{now}", now)
    splits_tpl = cfg.get("splits_path", "data/splits/splits_{now}.json")
    splits_path = splits_tpl.replace("{now}", now)
    return now, processed_dir, output_dir, splits_path

def normalize_ratios(train_r, val_r, test_r):
    s = train_r + val_r + test_r
    if s <= 0:
        return 0.8, 0.1, 0.1
    if abs(s - 1.0) < 1e-6:
        return train_r, val_r, test_r
    return train_r/s, val_r/s, test_r/s

def msss_split(X_paths, Y_bin, seed, train_r, val_r, test_r):
    holdout = val_r + test_r
    if holdout <= 0:
        return X_paths, [], []
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
    idx_train, idx_tmp = next(msss.split(np.zeros((len(X_paths), 1)), Y_bin))
    X_tmp, Y_tmp = [X_paths[i] for i in idx_tmp], Y_bin[idx_tmp]
    frac_test_in_tmp = test_r / (val_r + test_r) if (val_r + test_r) > 0 else 0.5
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=frac_test_in_tmp, random_state=seed)
    idx_val, idx_test = next(msss2.split(np.zeros((len(X_tmp), 1)), Y_tmp))
    return [X_paths[i] for i in idx_train], [X_tmp[i] for i in idx_val], [X_tmp[i] for i in idx_test]

def load_hist_from_selection(selected_json, num_classes):
    """Returns list of dicts with {input,label,zone,key,hist(np.ndarray)}"""
    with open(selected_json, "r") as f:
        sel = json.load(f)
    tiles = sel.get("tiles", [])
    if not tiles:
        raise RuntimeError(f"No tiles in {selected_json}")

    rows = []
    for rec in tiles:
        ip = rec.get("input")
        lp = rec.get("label")
        if not ip or not lp:
            continue
        if not (os.path.isfile(ip) and os.path.isfile(lp)):
            continue
        # hist may be saved as list or dict in selection JSON
        hist = rec.get("hist")
        if isinstance(hist, dict):
            # dict of class->count
            h = np.zeros(num_classes, dtype=np.int64)
            for k, v in hist.items():
                try:
                    ki = int(k)
                except Exception:
                    continue
                if 0 <= ki < num_classes:
                    h[ki] = int(v)
            hist = h
        elif isinstance(hist, list):
            arr = np.array(hist, dtype=np.int64)
            if arr.size < num_classes:
                pad = np.zeros(num_classes - arr.size, dtype=np.int64)
                arr = np.concatenate([arr, pad], axis=0)
            hist = arr[:num_classes]
        else:
            hist = None

        zone = rec.get("zone")
        if not zone:
            zone, _ = parse_zone_tile(os.path.basename(ip))
            zone = zone or "00000"

        rows.append({
            "input": ip,
            "label": lp,
            "zone": zone,
            "key": rec.get("key", f"{zone}_{os.path.splitext(os.path.basename(ip))[0].split('_')[-1]}"),
            "hist": hist
        })
    return rows

def load_hist_jsonl_fallback(jsonl_path, num_classes):
    """Optional fallback if selection JSON lacks per-tile hist; returns key->hist ndarray"""
    if not os.path.isfile(jsonl_path):
        return {}
    out = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = rec.get("key")
            hist = rec.get("hist")
            if key is None or hist is None:
                continue
            arr = np.array(hist, dtype=np.int64)
            if arr.size < num_classes:
                pad = np.zeros(num_classes - arr.size, dtype=np.int64)
                arr = np.concatenate([arr, pad], axis=0)
            out[key] = arr[:num_classes]
    return out

# -------------------------
# Core
# -------------------------
def generate_splits_and_weights(config, selected_json: str | None = None):
    num_classes = int(config.get("num_classes", 13))

    split_cfg = config.get("splitting", {})
    seed      = int(split_cfg.get("seed", 42))
    train_r   = float(split_cfg.get("train_ratio", 0.8))
    val_r     = float(split_cfg.get("val_ratio",   0.1))
    test_r    = float(split_cfg.get("test_ratio",  0.1))
    train_r, val_r, test_r = normalize_ratios(train_r, val_r, test_r)

    now, processed_dir, output_dir, splits_path = resolve_now_from_config(config)
    split_hash = get_split_hash(config)
    config["splits_path"] = splits_path  # persist for train.py

    # ---------- Load selection (with hist already stored) ----------
    if selected_json is None:
        selected_json = os.path.join("data", "splits", f"selected_{now}.json")
    if not os.path.isfile(selected_json):
        raise FileNotFoundError(f"Selected tiles JSON not found: {selected_json}")

    rows = load_hist_from_selection(selected_json, num_classes)

    # Optional fallback: fill missing hists from jsonl index
    missing = [r for r in rows if r["hist"] is None]
    if missing:
        idx = load_hist_jsonl_fallback(os.path.join("data", "labels", "label_histograms.jsonl"), num_classes)
        for r in missing:
            if r["key"] in idx:
                r["hist"] = idx[r["key"]]

    # Drop any still-missing hist
    rows = [r for r in rows if r["hist"] is not None and r["hist"].sum() > 0 and r["hist"][0] < r["hist"].sum()]
    if not rows:
        raise RuntimeError("No rows with valid histograms remain. Check selection JSON.")

    # ---------- Build multilabel presence from hist ----------
    X_all = [r["input"] for r in rows]
    Y_all = np.stack([(r["hist"] > 0).astype(np.int8) for r in rows], axis=0)
    Z_all = [r["zone"] for r in rows]
    pixel_hist_map = {r["input"]: r["hist"] for r in rows}

    # ---------- Split (optionally per zone) ----------
    stratify_by_zone = bool(split_cfg.get("stratify_by_utm", False))

    train_list, val_list, test_list = [], [], []
    if stratify_by_zone:
        zones = sorted(set(Z_all))
        print(f"[INFO] Splitting per zone across {len(zones)} zones…")
        for z in zones:
            idx = [i for i, zz in enumerate(Z_all) if zz == z]
            Xz  = [X_all[i] for i in idx]
            Yz  = Y_all[idx]
            if len(Xz) < 3:
                train_list.extend(Xz)
                continue
            tr, va, te = msss_split(Xz, Yz, seed=seed, train_r=train_r, val_r=val_r, test_r=test_r)
            train_list.extend(tr); val_list.extend(va); test_list.extend(te)
    else:
        tr, va, te = msss_split(X_all, Y_all, seed=seed, train_r=train_r, val_r=val_r, test_r=test_r)
        train_list, val_list, test_list = tr, va, te

    # ---------- Save splits JSON ----------
    splits = {
        "train": sorted(train_list),
        "val":   sorted(val_list),
        "test":  sorted(test_list),
    }
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[INFO] Saved splits → {splits_path}")
    print(f"       counts: train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")

    # ---------- Compute weights (from training hist only) ----------
    print("[INFO] Computing class & sample weights on training set…")
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    for ip in tqdm(splits["train"], ncols=80, desc="Class counts"):
        hist = pixel_hist_map.get(ip)
        if hist is not None:
            pixel_counts += hist

    # inverse-frequency with smoothing
    class_weights = 1.0 / (pixel_counts.astype(np.float64) + 1e-6)
    class_weights *= (num_classes / class_weights.sum())

    sample_weights = []
    for ip in splits["train"]:
        hist = pixel_hist_map.get(ip, np.zeros(num_classes, dtype=np.int64))
        present_idx = np.where(hist > 0)[0]
        w = 0.0 if present_idx.size == 0 else float(np.mean(class_weights[present_idx]))
        sample_weights.append(w)

    # ---------- Save weights ----------
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    now_tag = os.path.basename(os.path.normpath(processed_dir))
    np.savez(
        os.path.join(weights_dir, f"weights_{now_tag}_{get_split_hash(config)}.npz"),
        class_weights=class_weights.astype(np.float32),
        sample_weights=np.array(sample_weights, dtype=np.float32)
    )
    np.save(
        os.path.join(weights_dir, f"weights_{now_tag}_{get_split_hash(config)}_pixels.npy"),
        pixel_counts
    )
    print(f"[INFO] Saved weights → {weights_dir}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--selected", default=None,
                        help="Path to selected tiles JSON (default: data/splits/selected_{now}.json)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    generate_splits_and_weights(cfg, selected_json=args.selected)
