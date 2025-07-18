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

# ========== Load config ==========
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
    candidates = glob.glob(os.path.join(BASE_DIR, "outputs", "20*_*"))
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
def parse_best_epoch(log_path, fallback_txt_path=None):
    # First try best_epoch.txt if available
    if fallback_txt_path and os.path.exists(fallback_txt_path):
        with open(fallback_txt_path, "r") as f:
            line = f.readline().strip()
            if "," in line:
                epoch, f1 = line.split(",")
                return int(epoch), float(f1)

    # Fallback to parsing CSV-style training log
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
    model = ResNetUNetViT(n_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS).cuda()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model

def run_inference(model, input_tif, output_tif):
    with rasterio.open(input_tif) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype='uint8', compress='lzw')

        h, w = src.height, src.width
        pred_accum = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h, STRIDE):
            for x in range(0, w, STRIDE):
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                img = src.read(list(range(2, 2 + INPUT_CHANNELS)), window=window, boundless=True, fill_value=0).astype(np.float32)

                if img.shape[1] < PATCH_SIZE or img.shape[2] < PATCH_SIZE:
                    pad_img = np.zeros((img.shape[0], PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                    pad_img[:, :img.shape[1], :img.shape[2]] = img
                    img = pad_img

                img_tensor = torch.from_numpy(img).unsqueeze(0).cuda()
                with torch.no_grad():
                    logits = model(img_tensor).squeeze(0).cpu().numpy()  # [C, H, W]

                x0, y0 = x, y
                x1, y1 = x + PATCH_SIZE, y + PATCH_SIZE
                pred_accum[:, y0:y1, x0:x1] += logits[:, :min(h - y0, PATCH_SIZE), :min(w - x0, PATCH_SIZE)]
                weight_map[y0:y1, x0:x1] += 1

        # Normalize logits by weight map
        weight_map = np.clip(weight_map, a_min=1e-6, a_max=None)
        avg_logits = pred_accum / weight_map

        pred = np.argmax(avg_logits, axis=0).astype(np.uint8)
        decoded = np.vectorize(lambda x: inverse_remap.get(x, 255))(pred)

        with rasterio.open(output_tif, 'w', **meta) as dst:
            dst.write(decoded.astype(np.uint8), 1)


# ========== Main ==========
if __name__ == "__main__":
    best_epoch, best_f1 = parse_best_epoch(LOG_PATH, fallback_txt_path=os.path.join(CKPT_DIR, "best_epoch.txt"))
    if best_epoch is None:
        raise RuntimeError("No valid epoch found in training log.")

    ckpt_path = os.path.join(CKPT_DIR, f"model_epoch{best_epoch}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[WARNING] model_epoch{best_epoch}.pt not found. Trying best_model.pt...")
        best_path = os.path.join(CKPT_DIR, "best_model.pt")
        if os.path.exists(best_path):
            ckpt_path = best_path
        else:
            ckpt_files = glob.glob(os.path.join(CKPT_DIR, "model_epoch*.pt"))
            if ckpt_files:
                available_epochs = sorted([
                    int(re.search(r'model_epoch(\d+)\.pt', os.path.basename(f)).group(1))
                    for f in ckpt_files
                ])
                fallback_epoch = available_epochs[-1]
                ckpt_path = os.path.join(CKPT_DIR, f"model_epoch{fallback_epoch}.pt")
                print(f"[WARNING] Fallback: using model_epoch{fallback_epoch}.pt")
            else:
                raise FileNotFoundError(f"No model checkpoints available in {CKPT_DIR}")

    print(f"[INFO] Loading best model from epoch {best_epoch} with F1 score {best_f1:.4f}")
    model = load_model(ckpt_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_files = glob.glob(os.path.join(INPUT_DIR, "*.tif"))
    print(f"[INFO] Found {len(input_files)} input images in {INPUT_DIR}")
    print(f"[INFO] Saving predictions to {OUTPUT_DIR}")

    for input_path in tqdm(input_files, desc="Running inference", unit="tile"):
        filename = os.path.basename(input_path)
        output_path = os.path.join(OUTPUT_DIR, filename)
        run_inference(model, input_path, output_path)

    print("[INFO] Inference complete.")
