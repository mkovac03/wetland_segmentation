from glob import glob
import numpy as np
import os

paths = glob("data/processed/20250806_153626/*_img.npy")
for img_path in paths:
    base = img_path.replace("_img.npy", "")
    lbl_path = base + "_lbl.npy"
    try:
        img = np.load(img_path)
        lbl = np.load(lbl_path)
        if img.ndim != 3 or img.shape[0] != 22 or img.shape[1:] != lbl.shape:
            print(f"[BAD SHAPE] {base}: {img.shape}, {lbl.shape}")
    except Exception as e:
        print(f"[CORRUPT] {base}: {e}")
