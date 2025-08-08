# File: split_data.py
import os
import json
import yaml
import numpy as np
import hashlib
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from data.dataset import get_file_list

def get_split_hash(cfg):
    split_cfg = cfg["splitting"]
    hash_input = json.dumps({
        "train_ratio": split_cfg.get("train_ratio", 0.8),
        "val_ratio": split_cfg.get("val_ratio", 0.1),
        "test_ratio": split_cfg.get("test_ratio", 0.1),
        "seed": split_cfg.get("seed", 42),
        "num_classes": cfg["num_classes"]
    }, sort_keys=True)
    return hashlib.md5(hash_input.encode()).hexdigest()[:8]

def generate_splits_and_weights(config):
    input_dir = config["processed_dir"].rstrip("/")
    output_dir = config["output_dir"].rstrip("/")
    num_classes = config["num_classes"]
    split_hash = get_split_hash(config)
    timestamp = os.path.basename(input_dir)
    split_path = f"data/splits/splits_{timestamp}_{split_hash}.json"
    config["splits_path"] = split_path

    split_cfg = config.get("splitting", {})
    seed = split_cfg.get("seed", 42)
    train_ratio = split_cfg.get("train_ratio", 0.8)
    val_ratio = split_cfg.get("val_ratio", 0.1)
    test_ratio = split_cfg.get("test_ratio", 0.1)

    # ========== Find valid tiles ==========
    print("[INFO] Collecting label histograms and class presence...")
    X, Y = [], []

    hist_dir = os.path.join(input_dir, "label_histograms")
    tile_ids = set()

    for f in os.listdir(input_dir):
        if f.endswith("_lbl.npy"):
            tile_id = f.replace("_lbl.npy", "")
            img_path = os.path.join(input_dir, f"{tile_id}_img.npy")
            lbl_path = os.path.join(input_dir, f"{tile_id}_lbl.npy")
            hist_path = os.path.join(hist_dir, f"{tile_id}.json")
            if os.path.exists(img_path) and os.path.exists(lbl_path) and os.path.exists(hist_path):
                tile_ids.add(tile_id)

    print(f"[DEBUG] Valid tile count with img + lbl + hist: {len(tile_ids)}")

    for tile_id in tqdm(sorted(tile_ids), desc="Valid histogram tiles"):
        hist_path = os.path.join(hist_dir, f"{tile_id}.json")
        with open(hist_path) as f:
            data = json.load(f)
            hist = data.get("histogram", {})

        counts = np.zeros(num_classes, dtype=np.int64)
        for k, v in hist.items():
            try:
                k_int = int(k)
                if 0 <= k_int < num_classes:
                    counts[k_int] = v
            except ValueError:
                continue

        if counts.sum() == 0 or counts[0] == counts.sum():  # only background
            continue

        class_presence = (counts > 0).astype(int)
        X.append(os.path.join(input_dir, f"{tile_id}_img.npy"))
        Y.append(class_presence)

    print(f"[DEBUG] Total usable tiles: {len(X)}")

    if not Y:
        raise RuntimeError("No valid label histograms found. Check that tiles contain non-empty foreground labels.")

    X, Y = np.array(X), np.stack(Y)

    # ========== Stratified Splitting ==========
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio, random_state=seed)
    train_idx, temp_idx = next(msss.split(X, Y))
    X_temp, Y_temp = X[temp_idx], Y[temp_idx]
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)
    val_idx, test_idx = next(msss2.split(X_temp, Y_temp))

    splits = {
        "train": X[train_idx].tolist(),
        "val": X_temp[val_idx].tolist(),
        "test": X_temp[test_idx].tolist()
    }

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[INFO] Saved splits to {split_path}")

    # ========== Compute Weights ==========
    print("[INFO] Aggregating histograms from label_histograms...")
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    sample_weights = []
    for tile_path in tqdm(splits["train"], desc="Computing class/sample weights"):
        tile_id = os.path.basename(tile_path).replace("_img.npy", "")
        hist_path = os.path.join(hist_dir, f"{tile_id}.json")
        with open(hist_path) as f:
            data = json.load(f)
            hist = data.get("histogram", {})
        counts = np.zeros(num_classes, dtype=np.int64)
        for k, v in hist.items():
            try:
                k_int = int(k)
                if 0 <= k_int < num_classes:
                    counts[k_int] = v
            except ValueError:
                continue
        pixel_counts += counts

    class_weights = 1.0 / (pixel_counts + 1e-6)
    class_weights *= (num_classes / class_weights.sum())

    for tile_path in splits["train"]:
        tile_id = os.path.basename(tile_path).replace("_img.npy", "")
        hist_path = os.path.join(hist_dir, f"{tile_id}.json")
        with open(hist_path) as f:
            data = json.load(f)
            hist = data.get("histogram", {})
        counts = np.zeros(num_classes, dtype=np.int64)
        for k, v in hist.items():
            try:
                k_int = int(k)
                if 0 <= k_int < num_classes:
                    counts[k_int] = v
            except ValueError:
                continue
        present = np.where(counts > 0)[0]
        weights = class_weights[present] if len(present) > 0 else [0.0]
        sample_weights.append(float(np.mean(weights)))

    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    np.savez(os.path.join(weights_dir, f"weights_{timestamp}_{split_hash}.npz"),
             class_weights=class_weights, sample_weights=sample_weights)
    print(f"[INFO] Saved weights to {weights_dir}")

# ========== CLI ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    generate_splits_and_weights(cfg)
