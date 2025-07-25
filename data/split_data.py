import os
import json
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from data.dataset import get_file_list
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ========== Argument parsing ==========
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()

# ========== Load config ==========
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

INPUT_DIR  = config["processed_dir"].rstrip("/")
SPLIT_PATH = config["splits_path"]
NUM_CLASSES = config["num_classes"]

# ==== Splitting params from config ====
split_cfg = config.get("splitting", {})
BACKGROUND_CLASS = 0
IGNORE_INDEX = 255
BG_THRESHOLD = split_cfg.get("background_threshold", 0.7)
TESTVAL_RATIO = split_cfg.get("testval_ratio", 0.15)
VAL_RATIO_WITHIN_TESTVAL = split_cfg.get("val_ratio_within_testval", 0.5)
SEED = split_cfg.get("seed", 42)

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

# ========== Stratified Split ==========
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TESTVAL_RATIO, random_state=SEED)
train_idx, testval_idx = next(msss.split(X, Y))

X_testval = X[testval_idx]
Y_testval = Y[testval_idx]

msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO_WITHIN_TESTVAL, random_state=SEED)
val_idx, test_idx = next(msss2.split(X_testval, Y_testval))

splits = {
    "train": X[train_idx].tolist(),
    "val": X_testval[val_idx].tolist(),
    "test": X_testval[test_idx].tolist(),
}

# ========== Save ==========
os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
with open(SPLIT_PATH, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"[INFO] Saved splits to {SPLIT_PATH}")
print(f"[INFO] Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}, Total (after filtering): {len(X)}")
