# File: split_data.py
import os
import json
import yaml
import numpy as np
import torch
import hashlib
from tqdm import tqdm
from sklearn.utils import shuffle
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from data.dataset import get_file_list, GoogleEmbedDataset


def get_split_hash(cfg):
    split_cfg = cfg["splitting"]
    hash_input = json.dumps({
        "train_ratio": split_cfg.get("train_ratio", 0.8),
        "val_ratio": split_cfg.get("val_ratio", 0.1),
        "test_ratio": split_cfg.get("test_ratio", 0.1),
        "background_threshold": split_cfg.get("background_threshold", 0.9),
        "seed": split_cfg.get("seed", 42),
        "num_classes": cfg["num_classes"]
    }, sort_keys=True)
    return hashlib.md5(hash_input.encode()).hexdigest()[:8]


def generate_splits_and_weights(config):
    input_dir = config["processed_dir"].rstrip("/")
    num_classes = config["num_classes"]
    split_hash = get_split_hash(config)
    timestamp = os.path.basename(input_dir)
    split_path = f"data/splits/splits_{timestamp}_{split_hash}.json"
    config["splits_path"] = split_path

    split_cfg = config.get("splitting", {})
    background_class = 0
    ignore_index = 255
    bg_threshold = split_cfg.get("background_threshold", 0.7)
    seed = split_cfg.get("seed", 42)
    train_ratio = split_cfg.get("train_ratio", 0.8)
    val_ratio = split_cfg.get("val_ratio", 0.1)
    test_ratio = split_cfg.get("test_ratio", 0.1)
    max_tiles = split_cfg.get("max_training_tiles", None)
    entropy_cut = split_cfg.get("entropy_percentile", None)

    file_list = get_file_list(input_dir)
    if len(file_list) == 0:
        raise RuntimeError(f"No input .npy tiles found in {input_dir}")

    cache_path = os.path.join(input_dir, f"filtered_tiles_{num_classes}_classes.npz")
    if os.path.exists(cache_path):
        print(f"[INFO] Loading filtered tiles from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        X, Y = data["X"], data["Y"]
    else:
        print("[INFO] Filtering and analyzing label content for each tile...")
        X, Y = [], []
        for base in tqdm(file_list):
            lbl_path = base + "_lbl.npy"
            lbl = np.load(lbl_path)
            total = lbl.size
            bg_pixels = np.sum(lbl == background_class)
            ignore_pixels = np.sum(lbl == ignore_index)
            bg_ratio = bg_pixels / (total - ignore_pixels + 1e-6)
            if bg_ratio > bg_threshold:
                continue

            lbl_flat = lbl[lbl != ignore_index]
            counts = np.bincount(lbl_flat, minlength=num_classes)
            probs = counts / (counts.sum() + 1e-8)
            entropy = -np.sum(probs * np.log2(probs + 1e-8))

            class_presence = np.zeros(num_classes, dtype=int)
            for c in range(num_classes):
                if counts[c] > 0:
                    class_presence[c] = 1

            X.append((base, entropy))
            Y.append(class_presence)

        if len(X) == 0:
            raise RuntimeError("No tiles left after background filtering.")

        # Sort by entropy descending
        sorted_pairs = sorted(zip(X, Y), key=lambda pair: -pair[0][1])
        X_sorted, Y_sorted = zip(*sorted_pairs)
        X = np.array([x[0] for x in X_sorted])
        Y = np.stack(Y_sorted)

        if entropy_cut is not None:
            keep_n = int(len(X) * entropy_cut / 100)
            X = X[:keep_n]
            Y = Y[:keep_n]
            print(f"[INFO] Kept top {keep_n} tiles based on entropy percentile cutoff ({entropy_cut}%)")

        if max_tiles is not None and len(X) > max_tiles:
            X = X[:max_tiles]
            Y = Y[:max_tiles]
            print(f"[INFO] Truncated to max {max_tiles} tiles after entropy filtering")

        np.savez(cache_path, X=X, Y=Y)
        print(f"[INFO] Saved filtered tiles to cache: {cache_path}")

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio < 1.0:
        n_total = int(len(X) * total_ratio)
        X, Y = shuffle(X, Y, random_state=seed)
        X = X[:n_total]
        Y = Y[:n_total]
        print(f"[INFO] Subsampled {n_total} tiles based on total ratio {total_ratio:.2f}")

    testval_ratio = val_ratio + test_ratio
    val_ratio_within_temp = val_ratio / (val_ratio + test_ratio)

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=testval_ratio, random_state=seed)
    train_idx, temp_idx = next(msss.split(X, Y))
    X_temp, Y_temp = X[temp_idx], Y[temp_idx]
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 - val_ratio_within_temp, random_state=seed)
    val_idx, test_idx = next(msss2.split(X_temp, Y_temp))

    splits = {
        "train": X[train_idx].tolist(),
        "val": X_temp[val_idx].tolist(),
        "test": X_temp[test_idx].tolist(),
    }

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"[INFO] Saved splits to {split_path}")
    print(f"[INFO] Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}, Total: {len(X)}")

    return splits


# ========= CLI entry point ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    generate_splits_and_weights(cfg)
