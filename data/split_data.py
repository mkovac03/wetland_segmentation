import os
import json
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from data.dataset import get_file_list, GoogleEmbedDataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.utils import shuffle
import torch

# ========== Argument parsing ==========
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()

# ========== Load config ==========
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

INPUT_DIR = config["processed_dir"].rstrip("/")
SPLIT_PATH = config["splits_path"]
NUM_CLASSES = config["num_classes"]

# ==== Splitting params from config ====
split_cfg = config.get("splitting", {})
BACKGROUND_CLASS = 0
IGNORE_INDEX = 255
BG_THRESHOLD = split_cfg.get("background_threshold", 0.7)
SEED = split_cfg.get("seed", 42)
TRAIN_RATIO = split_cfg.get("train_ratio", 0.8)
VAL_RATIO = split_cfg.get("val_ratio", 0.1)
TEST_RATIO = split_cfg.get("test_ratio", 0.1)

# ========== Get file list ==========
file_list = get_file_list(INPUT_DIR)
if len(file_list) == 0:
    raise RuntimeError(f"No input .npy tiles found in {INPUT_DIR}")

X, Y = [], []

print("[INFO] Filtering and analyzing label content for each tile...")
for base in tqdm(file_list):
    lbl_path = base + "_lbl.npy"
    lbl = np.load(lbl_path)

    total = lbl.size
    bg_pixels = np.sum(lbl == BACKGROUND_CLASS)
    ignore_pixels = np.sum(lbl == IGNORE_INDEX)
    bg_ratio = bg_pixels / (total - ignore_pixels + 1e-6)

    if bg_ratio > BG_THRESHOLD:
        continue

    class_presence = np.zeros(NUM_CLASSES, dtype=int)
    for c in range(NUM_CLASSES):
        if np.any(lbl == c):
            class_presence[c] = 1

    X.append(base)
    Y.append(class_presence)

if len(X) == 0:
    raise RuntimeError("No tiles left after background filtering.")

X = np.array(X)
Y = np.stack(Y)

# ========== Limit total set based on train+val+test ratios ==========
total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
if total_ratio < 1.0:
    n_total = int(len(X) * total_ratio)
    X, Y = shuffle(X, Y, random_state=SEED)
    X = X[:n_total]
    Y = Y[:n_total]
    print(f"[INFO] Subsampled {n_total} tiles based on total ratio {total_ratio:.2f}")

# ========== Multilabel stratified splitting ==========
testval_ratio = VAL_RATIO + TEST_RATIO
val_ratio_within_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=testval_ratio, random_state=SEED)
train_idx, temp_idx = next(msss.split(X, Y))

X_temp = X[temp_idx]
Y_temp = Y[temp_idx]

msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 - val_ratio_within_temp, random_state=SEED)
val_idx, test_idx = next(msss2.split(X_temp, Y_temp))

splits = {
    "train": X[train_idx].tolist(),
    "val": X_temp[val_idx].tolist(),
    "test": X_temp[test_idx].tolist(),
}

# ========== Save splits ==========
os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
with open(SPLIT_PATH, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"[INFO] Saved splits to {SPLIT_PATH}")
print(f"[INFO] Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}, Total (after filtering): {len(X)}")

# ========== Pixel-wise class frequency computation ==========
print("[INFO] Calculating pixel-wise class distribution for weighting...")
train_ds = GoogleEmbedDataset(splits["train"], check_files=True)
pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

for i in tqdm(range(len(train_ds)), desc="Counting pixels"):
    _, lbl = train_ds[i]
    for c in range(NUM_CLASSES):
        pixel_counts[c] += torch.sum(lbl == c).item()

timestamp = os.path.basename(INPUT_DIR)
pixel_path = os.path.join(INPUT_DIR, f"weights_{timestamp}_pixels.npy")
os.makedirs(os.path.dirname(pixel_path), exist_ok=True)
np.save(pixel_path, pixel_counts)
print(f"[INFO] Saved pixel counts to {pixel_path}")
