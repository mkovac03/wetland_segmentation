import os
import random
import json
import yaml
from datetime import datetime
from data.dataset import get_file_list

import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Seed
random.seed(42)

# Paths
INPUT_DIR = config["processed_dir"]
SPLIT_PATH = config["splits_path"]

# Collect and shuffle files
file_list = get_file_list(INPUT_DIR)
random.shuffle(file_list)

n_total = len(file_list)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

splits = {
    "train": file_list[:n_train],
    "val": file_list[n_train:n_train+n_val],
    "test": file_list[n_train+n_val:]
}

# Save
os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
with open(SPLIT_PATH, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"Saved splits to {SPLIT_PATH}")
