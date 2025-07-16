# File: utils/visualize_predictions.py
import os
import glob
import random
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Paths
INPUT_DIR = "/media/lkm413/storage1/gee_embedding_download/images/Denmark/2018/"
BASE_PRED_DIR = "/outputs/"

# Find latest prediction folder
folders = sorted(glob.glob(os.path.join(BASE_PRED_DIR, "predictions_Denmark2018_*")), key=os.path.getmtime, reverse=True)
PRED_DIR = folders[0] if folders else None
if not PRED_DIR:
    raise RuntimeError("No prediction folders found.")

# Timestamp for outputs
timestamp = os.path.basename(PRED_DIR).split("_")[-1]
OUTPUT_FILE = f"label_prediction_comparison_{timestamp}.png"

# Parameters
NUM_SAMPLES = 10
NODATA_VALUE = 255

# === Utility ===
def read_first_band(path):
    with rasterio.open(path) as src:
        return src.read(1)

def plot_comparison(gt_img, pred_img, title_gt, title_pred, ax_gt, ax_pred):
    cmap = "tab20"
    ax_gt.imshow(np.where(gt_img == NODATA_VALUE, np.nan, gt_img), cmap=cmap, vmin=0, vmax=19)
    ax_gt.set_title(title_gt)
    ax_gt.axis('off')
    ax_pred.imshow(np.where(pred_img == NODATA_VALUE, np.nan, pred_img), cmap=cmap, vmin=0, vmax=19)
    ax_pred.set_title(title_pred)
    ax_pred.axis('off')

# === Main ===
def main():
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.tif")))

    common_files = sorted(set(os.path.basename(f) for f in input_files) & set(os.path.basename(f) for f in pred_files))
    if len(common_files) == 0:
        raise RuntimeError("No matching files found between input and prediction directories")

    selected_files = random.sample(common_files, min(NUM_SAMPLES, len(common_files)))

    fig, axes = plt.subplots(NUM_SAMPLES, 2, figsize=(10, 4 * NUM_SAMPLES))
    if NUM_SAMPLES == 1:
        axes = np.expand_dims(axes, 0)

    for i, fname in enumerate(selected_files):
        gt_img = read_first_band(os.path.join(INPUT_DIR, fname))
        pred_img = read_first_band(os.path.join(PRED_DIR, fname))
        plot_comparison(gt_img, pred_img, f"Ground Truth\n{fname}", f"Prediction\n{fname}", axes[i, 0], axes[i, 1])

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    plt.close()
    print(f"Saved comparison figure with {NUM_SAMPLES} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
