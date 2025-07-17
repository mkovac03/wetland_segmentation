import os
import random
import json
import yaml
import argparse
from data.dataset import get_file_list

# ========== Argument parsing ==========
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()

# ========== Load config ==========
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# ========== Seed for reproducibility ==========
random.seed(42)

# ========== Paths ==========
INPUT_DIR = config["processed_dir"].rstrip("/")
SPLIT_PATH = config["splits_path"]

# ========== Collect file list ==========
file_list = get_file_list(INPUT_DIR)
if len(file_list) == 0:
    raise RuntimeError(f"No input .npy tiles found in {INPUT_DIR}. Did you run preprocessing?")

random.shuffle(file_list)

# ========== Split ratios ==========
n_total = len(file_list)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

splits = {
    "train": file_list[:n_train],
    "val": file_list[n_train:n_train + n_val],
    "test": file_list[n_train + n_val:]
}

# ========== Save ==========
os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
with open(SPLIT_PATH, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"[INFO] Saved splits to {SPLIT_PATH}")
print(f"[INFO] Train: {n_train}, Val: {n_val}, Test: {n_test}, Total: {n_total}")
