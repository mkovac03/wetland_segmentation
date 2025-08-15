# File: predict/vis_preds_vs_labels.py
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ===== Hardcoded paths =====
PRED_DIR = "/home/lkm413/wetland_segmentation/outputs/20250806_153626/predictions"
PROCESSED_DIR = "/media/lkm413/storage11/wetland_segmentation/data/processed/20250806_153626"
MAX_SAMPLES = 10  # how many to display
NUM_CLASSES = 13  # adjust to your dataset

# ===== Make discrete colormap =====
tab20 = plt.cm.get_cmap("tab20", NUM_CLASSES)  # tab20 with NUM_CLASSES discrete colors
norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, NUM_CLASSES + 0.5, 1), ncolors=NUM_CLASSES)

# ===== Main =====
pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*_pred.npy")))
if not pred_files:
    raise FileNotFoundError(f"No *_pred.npy files found in {PRED_DIR}")

for idx, pred_path in enumerate(pred_files):
    if idx >= MAX_SAMPLES:
        break

    base_name = os.path.basename(pred_path).replace("_pred.npy", "")
    lbl_name = base_name + "_lbl.npy"
    lbl_path = os.path.join(PROCESSED_DIR, lbl_name)

    if not os.path.exists(lbl_path):
        print(f"[WARN] No label found for {base_name}, skipping...")
        continue

    pred = np.load(pred_path)
    lbl = np.load(lbl_path)

    if pred.shape != lbl.shape:
        print(f"[WARN] Shape mismatch for {base_name}: pred {pred.shape}, label {lbl.shape}")
        continue

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(pred, cmap=tab20, norm=norm, interpolation="nearest")
    axes[0].set_title("Prediction")
    axes[0].axis("off")

    axes[1].imshow(lbl, cmap=tab20, norm=norm, interpolation="nearest")
    axes[1].set_title("Label")
    axes[1].axis("off")

    plt.suptitle(base_name)
    plt.tight_layout()
    plt.show()
