import os
import glob
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import yaml
import re
from scipy.special import softmax
from models.resunet_vit import ResNetUNetViT
from data.dataset import get_file_list
import matplotlib.pyplot as plt
import collections

# ========== Load config ==========
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

PATCH_SIZE = config["patch_size"]
STRIDE = config["stride"]
NUM_CLASSES = config["num_classes"]
INPUT_CHANNELS = config["input_channels"]
PROCESSED_DIR = config["processed_dir"]
OUTPUT_DIR_TEMPLATE = config["output_dir"]

# ========== Argument Parser ==========
parser = argparse.ArgumentParser()
parser.add_argument("--timestamp", help="Run timestamp (e.g. 20250730_141251)")
args = parser.parse_args()

# ========== Resolve timestamp and paths ==========
if args.timestamp:
    timestamp = args.timestamp
else:
    raise ValueError("You must provide --timestamp")

processed_dir = PROCESSED_DIR.format(now=timestamp)
output_dir = OUTPUT_DIR_TEMPLATE.format(now=timestamp)
LOG_PATH = os.path.join(output_dir, "best_epoch.txt")
CKPT_DIR = output_dir
PRED_OUTPUT_DIR = os.path.join(output_dir, f"predictions_2018_{timestamp}")

# ========== Utilities ==========
def parse_best_epoch(txt_path):
    if not os.path.exists(txt_path):
        return None, None
    with open(txt_path, "r") as f:
        line = f.readline().strip()
        if "," in line:
            epoch, _ = line.split(",")
            pattern = f"model_ep{int(epoch)}_weights.pt"
            candidates = [f for f in os.listdir(CKPT_DIR) if f.startswith("model_") and f.endswith("_weights.pt")]
            for c in candidates:
                if pattern in c:
                    return int(epoch), os.path.join(CKPT_DIR, c)
    return None, None

def load_model(ckpt_path):
    model = ResNetUNetViT(config).cuda()
    state_dict = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def run_inference(model, input_npy_path, output_tif_path):
    import rasterio
    from rasterio.transform import from_origin

    img = np.load(input_npy_path)  # shape [C, H, W]

    # ========== Preprocess input ==========
    invalid_mask = img == -32768
    img[invalid_mask] = 0  # Replace invalid with 0 (safe default)

    h, w = img.shape[1], img.shape[2]
    print(f"[DEBUG] Input stats: shape={img.shape}, min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}")

    logit_accum = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    y_positions = list(range(0, h - PATCH_SIZE + 1, STRIDE))
    x_positions = list(range(0, w - PATCH_SIZE + 1, STRIDE))
    if (h - PATCH_SIZE) % STRIDE != 0:
        y_positions.append(h - PATCH_SIZE)
    if (w - PATCH_SIZE) % STRIDE != 0:
        x_positions.append(w - PATCH_SIZE)

    for y in y_positions:
        for x in x_positions:
            patch = img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            dy, dx = patch.shape[1], patch.shape[2]
            pad_bottom = PATCH_SIZE - dy
            pad_right = PATCH_SIZE - dx
            patch = np.pad(patch, ((0, 0), (0, pad_bottom), (0, pad_right)), mode='edge')

            valid_mask = (patch != 0).any(axis=0)
            if valid_mask.sum() == 0:
                continue

            img_tensor = torch.from_numpy(patch).unsqueeze(0).cuda()
            with torch.no_grad():
                logits = model(img_tensor).squeeze(0).cpu().numpy()

            logits = logits[:, :dy, :dx]
            logit_accum[:, y:y+dy, x:x+dx] += logits * valid_mask[None, :, :]
            weight_map[y:y+dy, x:x+dx] += valid_mask.astype(np.float32)

    avg_logits = logit_accum / np.clip(weight_map, 1e-6, None)
    avg_probs = softmax(np.moveaxis(avg_logits, 0, -1), axis=-1)
    pred = np.argmax(avg_probs, axis=-1).astype(np.uint8)
    pred[weight_map == 1e-6] = 255

    counts = collections.Counter(pred.flatten().tolist())
    print(f"[DEBUG] Class distribution in prediction: {dict(counts)}")

    tif_name = os.path.basename(input_npy_path).replace("_img.npy", ".tif")
    tif_guess = os.path.join(config["input_dir"], tif_name)

    if not os.path.exists(tif_guess):
        print(f"[WARNING] GeoTIFF not found at {tif_guess}, using dummy georef.")
        transform = from_origin(0, 0, 10, 10)
        crs = config.get("crs_target", "EPSG:3035")
    else:
        with rasterio.open(tif_guess) as src:
            transform = src.transform
            crs = src.crs

    meta = {
        'driver': 'GTiff',
        'height': pred.shape[0],
        'width': pred.shape[1],
        'count': 1,
        'dtype': 'uint8',
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',
        'nodata': 255
    }

    with rasterio.open(output_tif_path, 'w', **meta) as dst:
        dst.write(pred, 1)

# ========== Main ==========
if __name__ == "__main__":
    best_epoch, ckpt_path = parse_best_epoch(LOG_PATH)
    if best_epoch is None or ckpt_path is None:
        raise RuntimeError("No valid checkpoint found from best_epoch.txt.")

    model = load_model(ckpt_path)

    os.makedirs(PRED_OUTPUT_DIR, exist_ok=True)
    input_files = get_file_list(processed_dir)

    print(f"[INFO] Found {len(input_files)} input tiles in {processed_dir}")
    print(f"[INFO] Saving predictions to {PRED_OUTPUT_DIR}")

    for base_path in tqdm(input_files, desc="Running inference", unit="tile"):
        input_npy = base_path + "_img.npy"
        output_tif = os.path.join(PRED_OUTPUT_DIR, os.path.basename(base_path) + "_pred.tif")
        run_inference(model, input_npy, output_tif)

    print("[INFO] Inference complete.")
