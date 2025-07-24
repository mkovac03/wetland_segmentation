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

# ========== SAFETY CHECK ==========
remapped_class_ids = sorted(inverse_remap.keys())
print(f"[INFO] Inverse remap covers model class indices: {remapped_class_ids}")
if len(remapped_class_ids) != NUM_CLASSES or max(remapped_class_ids) != NUM_CLASSES - 1:
    raise ValueError(
        f"[ERROR] Inconsistent remapping: model expects {NUM_CLASSES} classes, "
        f"but inverse_remap has keys: {remapped_class_ids}"
    )


# ========== Load Human-Readable Class Names ==========
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
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model

def gaussian_weight_mask(h, w, sigma=None):
    """Generates a normalized 2D Gaussian weight mask"""
    if sigma is None:
        sigma = h // 6
    y, x = np.indices((h, w))
    center_y, center_x = h // 2, w // 2
    mask = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
    return mask / mask.max()


def run_inference(model, input_tif, output_tif):
    with rasterio.open(input_tif) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype='uint8', compress='lzw', nodata=255)

        h, w = src.height, src.width
        pred_accum = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        if STRIDE >= PATCH_SIZE:
            print(f"[WARNING] STRIDE ({STRIDE}) >= PATCH_SIZE ({PATCH_SIZE}) — may cause gaps.")
        else:
            overlap = 100 - (100 * STRIDE / PATCH_SIZE)
            # print(f"[INFO] Using stride = {STRIDE} ({overlap:.1f}% overlap)")

        # Precompute Gaussian weight mask
        gauss_mask = gaussian_weight_mask(PATCH_SIZE, PATCH_SIZE)

        for y in range(0, h, STRIDE):
            for x in range(0, w, STRIDE):
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                img = src.read(list(range(2, 2 + INPUT_CHANNELS)),
                               window=window,
                               boundless=True,
                               fill_value=None).astype(np.float32)

                # Handle out-of-bounds tiles (rasterio may clip them)
                c, h_tile, w_tile = img.shape
                if h_tile < PATCH_SIZE or w_tile < PATCH_SIZE:
                    pad_bottom = PATCH_SIZE - h_tile
                    pad_right = PATCH_SIZE - w_tile
                    img = np.pad(img, ((0, 0), (0, pad_bottom), (0, pad_right)), mode="edge")

                img_tensor = torch.from_numpy(img).unsqueeze(0).cuda()
                with torch.no_grad():
                    logits = model(img_tensor).squeeze(0).cpu()
                    probs = torch.nn.functional.softmax(logits, dim=0).numpy()

                y1 = min(y + PATCH_SIZE, h)
                x1 = min(x + PATCH_SIZE, w)
                dy, dx = y1 - y, x1 - x

                pred_accum[:, y:y1, x:x1] += probs[:, :dy, :dx] * gauss_mask[:dy, :dx]
                weight_map[y:y1, x:x1] += gauss_mask[:dy, :dx]

        weight_map = np.clip(weight_map, 1e-6, None)
        avg_probs = pred_accum / weight_map
        pred = np.argmax(avg_probs, axis=0).astype(np.uint8)

        decoded = pred
        decoded = np.clip(decoded, 0, 255).astype(np.uint8)

        if label_names:
            class_descriptions = [label_names.get(str(c), f"Class {c}") for c in range(NUM_CLASSES)]
            meta["descriptions"] = tuple(class_descriptions)

        with rasterio.open(output_tif, 'w', **meta) as dst:
            dst.write(decoded, 1)



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
