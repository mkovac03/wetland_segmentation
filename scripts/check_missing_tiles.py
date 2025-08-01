import os
import glob
import numpy as np

input_dir = "/media/lkm413/storage21/gee_embedding_download/images/Europe/2018/bands_01_22" # e.g., "data/raw"
processed_dir = "data/processed/20250730_141251"

input_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))

missing = []
corrupt = []

for tif_path in input_files:
    base = os.path.splitext(os.path.basename(tif_path))[0]
    img_path = os.path.join(processed_dir, f"{base}_img.npy")
    lbl_path = os.path.join(processed_dir, f"{base}_lbl.npy")

    if not os.path.exists(img_path) or not os.path.exists(lbl_path):
        missing.append(base)
        continue

    try:
        img = np.load(img_path)
        lbl = np.load(lbl_path)
        if img.ndim != 3 or lbl.ndim != 2:
            corrupt.append(base)
    except Exception as e:
        corrupt.append(base)

total = len(input_files)
print(f"\nTotal .tif input tiles:   {total}")
print(f"Processed correctly:      {total - len(missing) - len(corrupt)}")
print(f"Missing .npy pairs:       {len(missing)}")
print(f"Corrupted .npy files:     {len(corrupt)}\n")

if missing:
    print("❌ Missing:")
    for b in missing[:10]:
        print(f"  {b} (missing .npy)")

if corrupt:
    print("\n💥 Corrupted:")
    for b in corrupt[:10]:
        print(f"  {b} (load or shape issue)")
