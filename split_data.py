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

    file_list = get_file_list(input_dir)
    if len(file_list) == 0:
        raise RuntimeError(f"No input .npy tiles found in {input_dir}")

    print("[INFO] Collecting label histograms and class presence...")
    X, Y = [], []
    hist_dir = os.path.join(input_dir, "label_histograms")
    for base in tqdm(file_list, desc="Valid histogram tiles"):
        tile_name = os.path.basename(base)
        hist_path = os.path.join(hist_dir, f"tile_{tile_name}.json")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            counts = np.array(json.load(f))
        if counts.sum() == 0:
            continue
        class_presence = (counts > 0).astype(int)
        X.append(base)
        Y.append(class_presence)

    X, Y = np.array(X), np.stack(Y)

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

    print("[INFO] Aggregating histograms from label_histograms...")
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    sample_weights = []
    for tile in tqdm(splits["train"], desc="Computing class/sample weights"):
        hist_path = os.path.join(hist_dir, f"tile_{os.path.basename(tile)}.json")
        with open(hist_path) as f:
            counts = np.array(json.load(f))
        pixel_counts += counts

    class_weights = 1.0 / (pixel_counts + 1e-6)
    class_weights *= (num_classes / class_weights.sum())

    for tile in splits["train"]:
        hist_path = os.path.join(hist_dir, f"tile_{os.path.basename(tile)}.json")
        with open(hist_path) as f:
            counts = np.array(json.load(f))
        present = np.where(counts > 0)[0]
        weights = class_weights[present] if len(present) > 0 else [0.0]
        sample_weights.append(float(np.mean(weights)))

    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    np.savez(os.path.join(weights_dir, f"weights_{timestamp}_{split_hash}.npz"),
             class_weights=class_weights, sample_weights=sample_weights)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    generate_splits_and_weights(cfg)
