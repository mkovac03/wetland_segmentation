# File: inference_zone.py
import os
import glob
import json
import torch
import argparse
import numpy as np
import rasterio
from tqdm import tqdm
from rasterio.windows import Window
from models.resunet_vit import ResNetUNetViT
import yaml
import re
from scipy.special import softmax
from multiprocessing import Pool, cpu_count

# ========== Load config ==========
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

PATCH_SIZE = config["patch_size"]
STRIDE = config["stride"]
NUM_CLASSES = config["num_classes"]
INPUT_CHANNELS = config["input_channels"]
BASE_DIR = config.get("base_dir", "/media/lkm413/storage1/wetland_segmentation")

# ========== Argument Parser ==========
parser = argparse.ArgumentParser()
parser.add_argument("--timestamp", required=True, help="Run timestamp (e.g. 20250715_182612)")
parser.add_argument("--zone_id", required=True, help="EPSG zone ID (e.g., 32632)")
args = parser.parse_args()

zone_folder = f"epsg{args.zone_id}_{args.timestamp}"
INPUT_DIR = os.path.join(BASE_DIR, "data", "merged_tif", zone_folder)
CKPT_DIR = os.path.join(BASE_DIR, "outputs", zone_folder)
OUTPUT_DIR = os.path.join(BASE_DIR, f"outputs/predictions_{zone_folder}")
LOG_PATH = os.path.join(CKPT_DIR, "training_log.txt")

# ========== Load Label Remap ==========
with open("data/label_remap.json", "r") as f:
    inverse_remap = {v: int(k) for k, v in json.load(f).items()}

remapped_class_ids = sorted(inverse_remap.keys())
print(f"[INFO] Model class indices: {remapped_class_ids}")
if len(remapped_class_ids) != NUM_CLASSES or max(remapped_class_ids) != NUM_CLASSES - 1:
    raise ValueError(
        f"[ERROR] Inconsistent remapping: model expects {NUM_CLASSES} classes, "
        f"but inverse_remap has keys: {remapped_class_ids}"
    )

label_names = {}
try:
    with open("data/label_remap_longnames.json", "r") as f:
        label_names = json.load(f)
except FileNotFoundError:
    print("[WARNING] label_remap_longnames.json not found. Proceeding without label names.")

# ========== Utilities ==========
def parse_best_epoch(log_path, fallback_txt_path=None):
    if fallback_txt_path and os.path.exists(fallback_txt_path):
        with open(fallback_txt_path, "r") as f:
            line = f.readline().strip()
            if "," in line:
                epoch, f1 = line.split(",")
                return int(epoch), float(f1)

    best_f1 = -1
    best_epoch = None
    with open(log_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 5:
                try:
                    epoch_num = int(parts[0])
                    f1_score = float(parts[-1])
                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_epoch = epoch_num
                except:
                    continue
    return best_epoch, best_f1

def load_model(ckpt_path):
    model = ResNetUNetViT(config).cuda()
    state_dict = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def run_inference(args):
    model, input_tif, output_tif = args

    if os.path.exists(output_tif):
        return f"[SKIP] {os.path.basename(output_tif)} already exists."

    with rasterio.open(input_tif) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype='uint8', compress='lzw', nodata=255)

        h, w = src.height, src.width
        y_positions = list(range(0, h - PATCH_SIZE + 1, STRIDE))
        x_positions = list(range(0, w - PATCH_SIZE + 1, STRIDE))

        if (h - PATCH_SIZE) % STRIDE != 0:
            y_positions.append(h - PATCH_SIZE)
        if (w - PATCH_SIZE) % STRIDE != 0:
            x_positions.append(w - PATCH_SIZE)

        logit_accum = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for y in y_positions:
            for x in x_positions:
                y1, x1 = y, x
                y2 = y + PATCH_SIZE
                x2 = x + PATCH_SIZE
                dy, dx = y2 - y1, x2 - x1

                window = Window(x1, y1, dx, dy)
                band_offset = config.get("band_offset", 2)
                band_indices = list(range(band_offset, band_offset + INPUT_CHANNELS))
                img = src.read(band_indices, window=window).astype(np.float32)

                if img.shape[1] == 0 or img.shape[2] == 0:
                    continue

                pad_bottom = PATCH_SIZE - dy
                pad_right = PATCH_SIZE - dx
                img = np.pad(img, ((0, 0), (0, pad_bottom), (0, pad_right)), mode='edge')

                img_tensor = torch.from_numpy(img).unsqueeze(0).cuda()

                def apply_tta(model, img_tensor):
                    variants = [img_tensor,
                                torch.flip(img_tensor, dims=[3]),
                                torch.flip(img_tensor, dims=[2]),
                                torch.flip(img_tensor, dims=[2, 3])]

                    logits_sum = 0
                    for variant in variants:
                        with torch.no_grad():
                            out = model(variant)
                        if variant is not img_tensor:
                            if variant.shape != img_tensor.shape:
                                raise RuntimeError("TTA dimension mismatch")
                            if torch.equal(variant, torch.flip(img_tensor, dims=[3])):
                                out = torch.flip(out, dims=[3])
                            elif torch.equal(variant, torch.flip(img_tensor, dims=[2])):
                                out = torch.flip(out, dims=[2])
                            else:
                                out = torch.flip(out, dims=[2, 3])
                        logits_sum += out

                    return (logits_sum / len(variants)).squeeze(0).cpu().numpy()

                logits = apply_tta(model, img_tensor)

                logit_accum[:, y1:y2, x1:x2] += logits[:, :dy, :dx]
                weight_map[y1:y2, x1:x2] += 1.0

        weight_map = np.clip(weight_map, 1e-6, None)
        avg_logits = logit_accum / weight_map
        avg_probs = softmax(avg_logits, axis=0)
        pred = np.argmax(avg_probs, axis=0).astype(np.uint8)
        decoded = np.clip(pred, 0, 255).astype(np.uint8)
        decoded[weight_map == 1e-6] = 255

        if label_names:
            meta["descriptions"] = tuple(label_names.get(str(c), f"Class {c}") for c in range(NUM_CLASSES))

        with rasterio.open(output_tif, 'w', **meta) as dst:
            dst.write(decoded, 1)

    return f"[DONE] {os.path.basename(input_tif)}"

# ========== Main ==========
best_epoch, best_f1 = parse_best_epoch(LOG_PATH, fallback_txt_path=os.path.join(CKPT_DIR, "best_epoch.txt"))
if best_epoch is None:
    raise RuntimeError("No valid epoch found in training log.")

ckpt_path = os.path.join(CKPT_DIR, f"model_epoch{best_epoch}_weights.pt")
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(CKPT_DIR, "best_model_weights.pt")

print(f"[INFO] Loading best model from epoch {best_epoch} with F1 score {best_f1:.4f}")
model = load_model(ckpt_path)
os.makedirs(OUTPUT_DIR, exist_ok=True)

input_files = glob.glob(os.path.join(INPUT_DIR, "*.tif"))
print(f"[INFO] Found {len(input_files)} input images in {INPUT_DIR}")
print(f"[INFO] Saving predictions to {OUTPUT_DIR}")

args_list = []
for input_path in input_files:
    filename = os.path.basename(input_path)
    output_path = os.path.join(OUTPUT_DIR, filename)
    args_list.append((model, input_path, output_path))

with Pool(processes=min(cpu_count(), 6)) as pool:
    for msg in tqdm(pool.imap_unordered(run_inference, args_list), total=len(args_list)):
        if msg:
            print(msg)

print("[INFO] Inference complete.")
