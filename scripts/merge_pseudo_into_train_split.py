import json
import os
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--split", required=True, help="Path to original split JSON")
args = parser.parse_args()

SPLIT_PATH = args.split
PSEUDO_DIR = "data/pseudo"
OUT_SPLIT_PATH = SPLIT_PATH.replace(".json", "_withpseudo.json")

with open(SPLIT_PATH, "r") as f:
    splits = json.load(f)

pseudo_paths = sorted(glob(os.path.join(PSEUDO_DIR, "*_lbl.npy")))
pseudo_bases = [p.replace("_lbl.npy", "") for p in pseudo_paths]

original_train = set(splits["train"])
new_pseudo = [p for p in pseudo_bases if p not in original_train]

print(f"[INFO] Found {len(new_pseudo)} new pseudo-labeled tiles to add.")
splits["train"].extend(new_pseudo)

with open(OUT_SPLIT_PATH, "w") as f:
    json.dump(splits, f, indent=2)

print(f"[INFO] Updated split saved to {OUT_SPLIT_PATH}")
