import os
import glob
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import rasterio
from datetime import datetime
from glob import glob
from rasterio.windows import Window
import re
from models.resunet_vit import ResNetUNetViT
import yaml

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

PATCH_SIZE = config["patch_size"]
STRIDE = config["stride"]
NUM_CLASSES = config["num_classes"]
INPUT_CHANNELS = config["input_channels"]
BASE_DIR = config.get("base_dir", "/media/lkm413/storage1/wetland_segmentation")
INPUT_DIR = config["input_dir"]


# ========== Argument Parser ==========
parser = argparse.ArgumentParser()
parser.add_argument("--timestamp", help="Run timestamp (e.g. 20250715_182612)")
args = parser.parse_args()

# ========== Resolve timestamp ==========
if args.timestamp:
    timestamp = args.timestamp
else:
    print("[INFO] No --timestamp provided. Searching for latest valid run...")
    candidates = glob(os.path.join(BASE_DIR, "outputs", "20*_*"))
    timestamps = sorted([
        os.path.basename(p) for p in candidates
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "training_log.txt"))
    ])
    if not timestamps:
        raise FileNotFoundError("No timestamped run folder with training_log.txt found in outputs/")
    timestamp = timestamps[-1]
    print(f"[INFO] Using latest run: {timestamp}")

LOG_PATH = os.path.join(BASE_DIR, "outputs", timestamp, "training_log.txt")
CKPT_DIR = os.path.join(BASE_DIR, "outputs", timestamp)
OUTPUT_DIR = os.path.join(BASE_DIR, f"outputs/predictions_Denmark2018_{timestamp}")

# ========== Load Label Remap ==========
with open("data/label_remap.json", "r") as f:
    inverse_remap = {v: int(k) for k, v in json.load(f).items()}

# ========== Utilities ==========
def parse_best_epoch(log_path):
    best_f1 = -1
    best_epoch = None
    with open(log_path, "r") as f:
        for line in f:
            if line.startswith("Epoch"):
                parts = line.strip().split(",")
                epoch_num = int(parts[0].split()[1])
                f1_score = float(parts[-1].split(":")[1].strip())
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_epoch = epoch_num
    return best_epoch, best_f1

def load_model(ckpt_path):
    model = ResNetUNetViT(n_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS).cuda()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model

def run_inference(model, input_tif, output_tif):
    with rasterio.open(input_tif) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype='uint8')
        if 'nodata' in meta and (meta['nodata'] is None or not (0 <= meta['nodata'] <= 255)):
            meta['nodata'] = None

        with rasterio.open(output_tif, 'w', **meta) as dst:
            for y in range(0, src.height, STRIDE):
                for x in range(0, src.width, STRIDE):
                    window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                    img = src.read(list(range(2, 2 + INPUT_CHANNELS)), window=window).astype(np.float32)

                    if img.shape[1] < PATCH_SIZE or img.shape[2] < PATCH_SIZE:
                        continue

                    img_tensor = torch.from_numpy(img).unsqueeze(0).cuda()
                    with torch.no_grad():
                        pred = model(img_tensor).argmax(1).squeeze().cpu().numpy()

                    decoded = np.vectorize(lambda x: inverse_remap.get(x, 255))(pred).astype("uint8")
                    dst.write(decoded, 1, window=window)

# ========== Main ==========
if __name__ == "__main__":
    best_epoch, best_f1 = parse_best_epoch(LOG_PATH)
    if best_epoch is None:
        raise RuntimeError("No valid epoch found in training log.")

    ckpt_path = os.path.join(CKPT_DIR, f"model_epoch{best_epoch}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")

    print(f"[INFO] Loading best model from epoch {best_epoch} with F1 score {best_f1:.4f}")
    model = load_model(ckpt_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_files = glob(os.path.join(INPUT_DIR, "*.tif"))
    print(f"[INFO] Found {len(input_files)} input images in {INPUT_DIR}")
    print(f"[INFO] Saving predictions to {OUTPUT_DIR}")

    for input_path in tqdm(input_files, desc="Running inference", unit="tile"):
        filename = os.path.basename(input_path)
        output_path = os.path.join(OUTPUT_DIR, filename)
        run_inference(model, input_path, output_path)

    print("[INFO] Inference complete.")
