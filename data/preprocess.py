# File: data/preprocess.py
import os
import glob
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from tqdm import tqdm
import argparse
import yaml

# ========== Utility ==========
def center_crop(array, target_size=(512, 512)):
    if array.ndim == 3:  # [C, H, W]
        _, h, w = array.shape
        th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return array[:, i:i+th, j:j+tw]
    elif array.ndim == 2:  # [H, W]
        h, w = array.shape
        th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return array[i:i+th, j:j+tw]
    else:
        raise ValueError(f"Unsupported array shape for cropping: {array.shape}")

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

INPUT_DIR = config["input_dir"]
OUTPUT_DIR = config["processed_dir"]
TARGET_CRS = config["crs_target"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving processed outputs to {OUTPUT_DIR}")

files = glob.glob(os.path.join(INPUT_DIR, '*.tif'))

for f in tqdm(files):
    needs_crop = False

    with rasterio.open(f) as src:
        if src.crs.to_string() != TARGET_CRS:
            dst_path = os.path.join(OUTPUT_DIR, os.path.basename(f).replace('.tif', '_reproj.tif'))
            transform, width, height = calculate_default_transform(
                src.crs, TARGET_CRS, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': TARGET_CRS,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(dst_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    resampling = Resampling.nearest if i == 1 else Resampling.bilinear
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=TARGET_CRS,
                        resampling=resampling
                    )
            out_path = dst_path
            needs_crop = True
        else:
            out_path = f

    with rasterio.open(out_path) as src:
        # Read label first
        label = src.read(1)

        # Replace NoData
        nodata_val = -32768
        ignore_val = 255
        label[label == nodata_val] = ignore_val

        # Define sparse → dense remap
        valid_classes = [0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        remap_dict = {old: new for new, old in enumerate(valid_classes)}
        inverse_remap = {v: k for k, v in remap_dict.items()}
        num_classes = len(remap_dict)

        # Apply remapping
        remapped_label = np.full_like(label, ignore_val, dtype=np.uint8)
        for old, new in remap_dict.items():
            remapped_label[label == old] = new
        label = remapped_label

        # Save remap dict once
        if not os.path.exists("data/label_remap.json"):
            os.makedirs("data", exist_ok=True)
            import json

            with open("data/label_remap.json", "w") as f:
                json.dump(remap_dict, f)
            print("Saved label remap to data/label_remap.json")

        # Read image *after* label handling
        if src.count < config["input_channels"] + 1:
            print(f"Skipping {out_path}, has only {src.count} bands")
            continue
        image = src.read(list(range(2, 2 + config["input_channels"]))).astype(np.float32)

        # Crop if needed
        if needs_crop:
            if image.shape[1:] != (512, 512):
                image = center_crop(image, (512, 512))
            if label.shape != (512, 512):
                label = center_crop(label, (512, 512))

        # Optional: mask image where label is ignore
        mask = label != ignore_val
        image = image * mask[None, :, :]

        # Final assert to catch broken tiles
        assert np.all(
            (label == ignore_val) | ((label >= 0) & (label < num_classes))), f"{out_path}: Label out of bounds"

        base = os.path.splitext(os.path.basename(out_path))[0].replace('_reproj', '')
        np.save(os.path.join(OUTPUT_DIR, f"{base}_img.npy"), image)
        np.save(os.path.join(OUTPUT_DIR, f"{base}_lbl.npy"), label)

