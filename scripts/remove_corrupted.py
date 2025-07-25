# File: delete_corrupt_tiles.py
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 🔧 Set your target directory here
DIR = "../data/processed/20250724_131236"

def is_corrupt(lbl_file):
    try:
        np.load(lbl_file)
        return None  # file is fine
    except Exception:
        base = lbl_file.replace("_lbl.npy", "")
        return base  # base path without suffix

if __name__ == "__main__":
    lbl_files = [os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith("_lbl.npy")]

    with Pool(max(cpu_count() - 1, 1)) as pool:
        results = list(tqdm(pool.imap_unordered(is_corrupt, lbl_files), total=len(lbl_files)))

    corrupted = [r for r in results if r is not None]

    print(f"Found {len(corrupted)} corrupted tiles.")
    for base in corrupted:
        img_path = base + "_img.npy"
        lbl_path = base + "_lbl.npy"
        try:
            os.remove(img_path)
            os.remove(lbl_path)
            print(f"Deleted: {os.path.basename(img_path)} and {os.path.basename(lbl_path)}")
        except FileNotFoundError:
            pass
