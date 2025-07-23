# File: data/preprocess.py
import os
import glob
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import json

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

# ========== Config & Args ==========
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

INPUT_DIR = config["input_dir"]
OUTPUT_DIR = config["processed_dir"]
TARGET_CRS = config["crs_target"]
INPUT_CHANNELS = config["input_channels"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving processed outputs to {OUTPUT_DIR}")

files = glob.glob(os.path.join(INPUT_DIR, '*.tif'))

# ========== Processing Loop ==========
for f in tqdm(files):
    needs_crop = False

    with rasterio.open(f) as src:
        # Reproject if needed
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
        label = src.read(1).astype(np.int32)

        # Replace NoData
        nodata_val = -32768
        ignore_val = 255
        label[label == nodata_val] = ignore_val

        # ========== Merge and remap labels ==========
        ignore_val = 255
        nodata_val = -32768
        label[label == nodata_val] = ignore_val

        # Merge classes based on specification
        merge_map = {
            0: [0],
            1: [1],
            2: [2, 4, 6],
            3: [8],
            4: [9],
            5: [10],
            6: [11],
            7: [12],
            8: [13],
            9: [14],
            10: [15],
            11: [16, 17, 20, 21, 22],
            12: [18],
            13: [19]
        }

        # Flatten mapping: reverse-lookup from original ID → remapped ID
        remap_dict = {}
        for new_id, old_ids in merge_map.items():
            for old_id in old_ids:
                remap_dict[old_id] = new_id
        num_classes = len(merge_map)

        # Apply remapping
        # Vectorized remap: fallback to 255 (ignore) if unmapped
        remapped_label = np.vectorize(lambda v: remap_dict.get(v, ignore_val))(label).astype(np.uint8)
        label = remapped_label

        # ========== Post-remap: check for unexpected label values ==========
        valid_ids = set(range(len(merge_map)))  # e.g. 0 to 13
        observed_ids = set(np.unique(label[label != ignore_val]))
        unexpected_ids = observed_ids - valid_ids

        if unexpected_ids:
            raise ValueError(
                f"[ERROR] Tile {os.path.basename(f)} contains unexpected remapped label(s): {sorted(unexpected_ids)}.\n"
                f"Check remap_dict and raw label values."
            )

        # Optional: skip mostly-background tiles
        background_label = remap_dict.get(0, None)
        if background_label is not None:
            background_ratio = np.sum(label == background_label) / label.size
            if background_ratio > 0.95:
                continue

        # Also save long name mapping
        label_names = {
            0: "No Wetland",
            1: "Rice Fields",
            2: "Riparian, fluvial and swamp forest (broadleaved, coniferous, mixed)",
            3: "Managed or grazed wet meadow or pasture",
            4: "Natural seasonally or permanently wet grasslands",
            5: "Wet heaths",
            6: "Riverine and fen scrubs",
            7: "Beaches, dunes, sand",
            8: "Inland marshes",
            9: "Open mires",
            10: "Salt marshes",
            11: "Surface water (lagoons, estuaries, rivers, lakes, shallow marine waters)",
            12: "Coastal saltpans (highly artificial salinas)",
            13: "Intertidal flats"
        }

        os.makedirs("data", exist_ok=True)
        with open("data/label_remap.json", "w") as f:
            json.dump(remap_dict, f)
        # print("Saved label remap to data/label_remap.json")

        with open("data/label_remap_longnames.json", "w") as f:
            json.dump(label_names, f, indent=2)
        # print("Saved long label names to data/label_remap_longnames.json")

        # Read image *after* label handling
        if src.count < INPUT_CHANNELS + 1:
            continue
        image = src.read(list(range(2, 2 + INPUT_CHANNELS))).astype(np.float32)

        # Crop if needed
        if needs_crop:
            if image.shape[1:] != (512, 512):
                image = center_crop(image, (512, 512))
            if label.shape != (512, 512):
                label = center_crop(label, (512, 512))

        # Mask ignore labels
        mask = label != ignore_val
        image = image * mask[None, :, :]

        # Validate label integrity
        assert np.all(
            (label == ignore_val) | ((label >= 0) & (label < num_classes))), f"{out_path}: Label out of bounds"

        # Save
        base = os.path.splitext(os.path.basename(out_path))[0].replace('_reproj', '')
        np.save(os.path.join(OUTPUT_DIR, f"{base}_img.npy"), image)
        np.save(os.path.join(OUTPUT_DIR, f"{base}_lbl.npy"), label)
