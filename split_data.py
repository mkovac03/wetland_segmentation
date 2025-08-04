# File: split_data.py
import os
import json
import yaml
import numpy as np
import torch
import hashlib
import re
from tqdm import tqdm
from sklearn.utils import shuffle
from collections import defaultdict
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
        "num_classes": cfg["num_classes"],
        "max_training_tiles": split_cfg.get("max_training_tiles", None),
        "entropy_percentile": split_cfg.get("entropy_percentile", None),
        "stratify_by_utm": split_cfg.get("stratify_by_utm", False),
        "utm_sampling_strategy": split_cfg.get("utm_sampling_strategy", "equal")
    }, sort_keys=True)
    return hashlib.md5(hash_input.encode()).hexdigest()[:8]

def extract_utm_zone(path):
    match = re.search(r"326\d{2}", path)
    return match.group(0) if match else "unknown"

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
    stratify_by_utm = split_cfg.get("stratify_by_utm", False)
    utm_strategy = split_cfg.get("utm_sampling_strategy", "equal")

    file_list = get_file_list(input_dir)
    if len(file_list) == 0:
        raise RuntimeError(f"No input .npy tiles found in {input_dir}")

    # Try to load cached filtered file list
    intermediate_path = os.path.join(input_dir, f"filtered_{timestamp}_{split_hash}.json")
    if os.path.exists(intermediate_path):
        print(f"[INFO] Found cached filtered tile list: {intermediate_path}. Loading...")
        with open(intermediate_path, "r") as f:
            filtered_info = json.load(f)
        X_full = filtered_info["X_full"]
        Y_full = [np.array(y) for y in filtered_info["Y_full"]]
        E_full = filtered_info["E_full"]
        U_full = filtered_info["U_full"]
    else:
        print("[INFO] Filtering and analyzing label content for each tile...")
        X_full, Y_full, E_full, U_full = [], [], [], []
        for base in tqdm(file_list):
            lbl_path = base + "_lbl.npy"
            lbl = np.load(lbl_path)
            total = lbl.size
            bg_pixels = np.sum(lbl == background_class)
            ignore_pixels = np.sum(lbl == ignore_index)
            bg_ratio = bg_pixels / (total - ignore_pixels + 1e-6)
            water_pixels = np.sum(lbl == 10)
            water_ratio = water_pixels / (total - ignore_pixels + 1e-6)

            if bg_ratio > bg_threshold or water_ratio > 0.9:
                continue

            lbl_flat = lbl[lbl != ignore_index]
            counts = np.bincount(lbl_flat, minlength=num_classes)
            probs = counts / (counts.sum() + 1e-8)
            entropy = -np.sum(probs * np.log2(probs + 1e-8))

            class_presence = np.zeros(num_classes, dtype=int)
            for c in range(num_classes):
                if counts[c] > 0:
                    class_presence[c] = 1

            utm = extract_utm_zone(base)
            X_full.append(base)
            Y_full.append(class_presence)
            E_full.append(entropy)
            U_full.append(utm)

        if len(X_full) == 0:
            raise RuntimeError("No tiles left after background filtering.")

        # Save intermediate
        filtered_info = {
            "X_full": X_full,
            "Y_full": [y.tolist() for y in Y_full],
            "E_full": E_full,
            "U_full": U_full
        }
        with open(intermediate_path, "w") as f:
            json.dump(filtered_info, f, indent=2)
        print(f"[INFO] Saved filtered tile info to {intermediate_path}")

    tiles_by_utm = defaultdict(list)
    for base, ent, cls, utm in zip(X_full, E_full, Y_full, U_full):
        tiles_by_utm[utm].append((base, ent, cls))

    selected_X, selected_Y = [], []
    utms_used = sorted(tiles_by_utm.keys())
    per_zone = max_tiles // len(utms_used) if max_tiles and utm_strategy == "equal" else None

    for utm in utms_used:
        zone_tiles = sorted(tiles_by_utm[utm], key=lambda t: -t[1])  # sort by entropy desc
        if entropy_cut:
            top_n = int(len(zone_tiles) * entropy_cut / 100)
            zone_tiles = zone_tiles[:top_n]

        if max_tiles:
            if utm_strategy == "equal":
                zone_tiles = zone_tiles[:per_zone]
            elif utm_strategy == "proportional":
                prop = len(zone_tiles) / len(X_full)
                zone_tiles = zone_tiles[:int(prop * max_tiles)]

        for base, _, cls in zone_tiles:
            selected_X.append(base)
            selected_Y.append(cls)

    X = np.array(selected_X)
    Y = np.stack(selected_Y)
    print(f"[INFO] Total selected tiles: {len(X)} after entropy and UTM-based filtering")

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

    print("[INFO] Calculating pixel-wise class distribution for weighting...")
    train_ds = GoogleEmbedDataset(splits["train"], check_files=True)
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    for i in tqdm(range(len(train_ds)), desc="Counting pixels"):
        _, lbl = train_ds[i]
        for c in range(num_classes):
            pixel_counts[c] += torch.sum(lbl == c).item()

    pixel_path = os.path.join(input_dir, f"weights_{timestamp}_{split_hash}_pixels.npy")
    np.save(pixel_path, pixel_counts)
    print(f"[INFO] Saved pixel counts to {pixel_path}")
    return splits, pixel_counts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    generate_splits_and_weights(cfg)
