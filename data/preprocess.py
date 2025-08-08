# File: data/preprocess.py (Updated to save + reload histograms before processing)
import os
import glob
import re
import json
import argparse
import yaml
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from collections import defaultdict
from scipy.stats import entropy

# ========== Utilities ==========

def extract_zone_and_index(path):
    basename = os.path.basename(path)
    zone_match = re.search(r'_(326\d{2})_', basename)
    index_match = re.findall(r"(\d+)(?=\.tif$)", basename)
    if zone_match and index_match:
        return zone_match.group(1), index_match[-1]
    return None, None

def center_crop(array, target_size=(512, 512)):
    if array.ndim == 3:
        _, h, w = array.shape
        th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return array[:, i:i+th, j:j+tw]
    elif array.ndim == 2:
        h, w = array.shape
        th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return array[i:i+th, j:j+tw]
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")

def reproject_stack(src, target_crs):
    dst_transform, width, height = calculate_default_transform(
        src.crs, target_crs, src.width, src.height, *src.bounds)
    out = np.empty((src.count, height, width), dtype=src.dtypes[0])
    for i in range(1, src.count + 1):
        reproject(
            source=rasterio.band(src, i),
            destination=out[i - 1],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest if i == 1 else Resampling.bilinear
        )
    return out, dst_transform

def compute_histogram(args):
    tile_key, tif_list, config = args
    try:
        target_crs = config["crs_target"]
        processed_dir = config["processed_dir"]
        remap_dict = config["label_remap"]
        ignore_val = config["ignore_val"]
        nodata_val = config["nodata_val"]

        hist_dir = os.path.join(processed_dir, "label_histograms")
        os.makedirs(hist_dir, exist_ok=True)

        label_path = sorted(tif_list)[0]  # assume label is in band 1 of the first tif

        with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
            with rasterio.open(label_path) as src:
                if src.crs == target_crs:
                    label = src.read(1)
                else:
                    reprojected, _ = reproject_stack(src, target_crs)
                    label = reprojected[0]

        label = label.astype(np.int32)
        label[label == nodata_val] = ignore_val
        remapped = np.vectorize(lambda v: remap_dict.get(v, ignore_val))(label).astype(np.uint8)

        classes, counts = np.unique(remapped[remapped != ignore_val], return_counts=True)
        hist = dict(zip(map(str, classes.tolist()), counts.tolist()))

        out_path = os.path.join(hist_dir, f"tile_{tile_key}.json")
        with open(out_path, "w") as f:
            json.dump({"tile_id": tile_key, "histogram": hist}, f)

        return tile_key, hist, remapped.shape[0], remapped.shape[1]

    except Exception as e:
        return tile_key, {"error": str(e)}, 0, 0


def should_process(hist, shape, threshold):
    if "error" in hist:
        return False

    total = shape[0] * shape[1]
    background = hist.get("0", 0)
    water = hist.get("10", 0)

    background_ratio = background / total
    water_ratio = water / total

    if background_ratio > threshold:
        return False
    if water_ratio > 0.5:
        return False

    return True


def label_entropy(hist):
    values = np.array(list(hist.values()))
    prob = values / values.sum()
    return entropy(prob)

def process_tile(args):
    tile_key, tif_list, config = args
    try:
        target_crs = config["crs_target"]
        patch_size = config.get("patch_size", 512)
        processed_dir = config["processed_dir"]
        remap_dict = config["label_remap"]
        ignore_val = config["ignore_val"]
        nodata_val = config["nodata_val"]
        expected_channels = config["input_channels"]
        num_classes = config["num_classes"]

        tif_list = sorted(tif_list)
        bands = []
        for path in tif_list:
            with rasterio.open(path) as src:
                array, _ = reproject_stack(src, target_crs)
                bands.extend(array)

        image = np.stack(bands)  # [C, H, W]
        label = image[0].astype(np.int32)
        label[label == nodata_val] = ignore_val
        remapped_label = np.vectorize(lambda v: remap_dict.get(v, ignore_val))(label).astype(np.uint8)

        # Check for presence of valid classes
        unique_vals = np.unique(remapped_label)
        if not np.any((unique_vals < num_classes) & (unique_vals != ignore_val)):
            return f"Tile {tile_key} skipped: no valid labels (only ignore or out of bounds)"

        # Final shape check
        if remapped_label.shape[0] < patch_size or remapped_label.shape[1] < patch_size:
            return f"Tile {tile_key} skipped: shape too small {remapped_label.shape}"

        remapped_label = center_crop(remapped_label, target_size=(patch_size, patch_size))

        # Remove label, drop bands 22 and 45 (indexing from 0, so 21 and 44)
        image_data = image[1:]  # [C-1, H, W]
        drop_idx = {22, 45}
        keep_bands = [i for i in range(image_data.shape[0]) if i not in drop_idx]
        filtered_image = image_data[keep_bands]

        # ✅ Now check channel count AFTER removing unwanted bands
        if filtered_image.shape[0] != config["input_channels"]:
            return f"Tile {tile_key} skipped: Expected {config['input_channels']} channels, got {filtered_image.shape[0]}"

        # Continue processing
        cropped_image = center_crop(filtered_image, target_size=(patch_size, patch_size))
        cropped_image = np.where(cropped_image == nodata_val, ignore_val, cropped_image).astype(filtered_image.dtype)

        np.save(os.path.join(processed_dir, f"tile_{tile_key}_lbl.npy"), remapped_label)
        np.save(os.path.join(processed_dir, f"tile_{tile_key}_img.npy"), cropped_image)

    except Exception as e:
        return f"[ERROR] Failed on tile {tile_key}: {e}"

    return None


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    input_dirs = config["input_dirs"]
    processed_dir = config["processed_dir"]
    splitting_cfg = config.get("splitting", {})
    bg_thresh = splitting_cfg.get("background_threshold", 0.9)
    max_tiles = splitting_cfg.get("max_training_tiles", 5000)
    strategy = splitting_cfg.get("utm_sampling_strategy", "equal")
    os.makedirs(processed_dir, exist_ok=True)

    all_tifs = []
    for d in input_dirs:
        all_tifs.extend(glob.glob(os.path.join(d, "*.tif")))

    tile_groups = {}
    tile_zones = {}
    for tif_path in all_tifs:
        zone, tile_id = extract_zone_and_index(tif_path)
        if zone and tile_id:
            key = f"{zone}_{tile_id}"
            tile_groups.setdefault(key, []).append(tif_path)
            tile_zones[key] = zone

    print(f"[INFO] Found {len(tile_groups)} tiles")
    print(f"[DEBUG] Zone counts: {dict((z, len([k for k in tile_zones if tile_zones[k] == z])) for z in set(tile_zones.values()))}")

    # ========== STEP 1: Compute and Save Histograms ==========
    hist_dir = os.path.join(processed_dir, "label_histograms")
    os.makedirs(hist_dir, exist_ok=True)

    args_list = []
    for tile_key, tif_list in tile_groups.items():
        json_path = os.path.join(hist_dir, f"tile_{tile_key}.json")
        if not os.path.exists(json_path):
            args_list.append((tile_key, tif_list, config))

    with Pool(max(1, cpu_count() - 1)) as pool:
        list(tqdm(pool.imap_unordered(compute_histogram, args_list, chunksize=32),
                  total=len(args_list), desc="Computing histograms"))

    # ========== STEP 2: Reload Histograms and Select Tiles ==========
    results = []
    for tile_key in tile_groups:
        json_path = os.path.join(hist_dir, f"tile_{tile_key}.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                entry = json.load(f)
                hist = entry.get("histogram", {})
                results.append((tile_key, hist, 512, 512))  # assumes 512x512

    # First apply background filter only
    bg_pass_tiles = [(tile_key, hist, h, w) for tile_key, hist, h, w in results
                     if "error" not in hist and (hist.get("0", 0) / (h * w)) < bg_thresh]

    print(f"[INFO] {len(bg_pass_tiles)} tiles pass background threshold < {bg_thresh}")

    # Then apply water filter on top
    valid_tiles = [(tile_key, hist) for tile_key, hist, h, w in bg_pass_tiles
                   if (hist.get("10", 0) / (h * w)) <= 0.5]

    num_water_excluded = len(bg_pass_tiles) - len(valid_tiles)
    print(f"[INFO] {num_water_excluded} tiles excluded due to >50% surface water (class 10)")

    zone_dict = defaultdict(list)
    for tile_key, hist in valid_tiles:
        zone = tile_zones[tile_key]
        zone_dict[zone].append((tile_key, hist))

    selected = []
    if strategy == "equal":
        per_zone = max_tiles // len(zone_dict)
        for zone, tiles in zone_dict.items():
            sorted_tiles = sorted(tiles, key=lambda x: label_entropy(x[1]), reverse=True)
            selected.extend(sorted_tiles[:per_zone])
    else:
        total_valid = sum(len(v) for v in zone_dict.values())
        for zone, tiles in zone_dict.items():
            share = int(max_tiles * len(tiles) / total_valid)
            sorted_tiles = sorted(tiles, key=lambda x: label_entropy(x[1]), reverse=True)
            selected.extend(sorted_tiles[:share])

    print(f"[INFO] Selected {len(selected)} tiles for processing")

    # ========== DEBUG: Zone distribution in selected tiles ==========
    zone_counts = defaultdict(int)
    for tile_key, _ in selected:
        zone = tile_key.split("_")[0]
        zone_counts[zone] += 1

    print("[DEBUG] Selected tile counts per zone:")
    for zone, count in sorted(zone_counts.items()):
        print(f"  Zone {zone}: {count}")

    # ========== STEP 3: Process Selected Tiles ==========
    process_args = []
    for tile_key, _ in selected:
        img_path = os.path.join(processed_dir, f"tile_{tile_key}_img.npy")
        lbl_path = os.path.join(processed_dir, f"tile_{tile_key}_lbl.npy")

        needs_processing = False
        if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
            needs_processing = True
        else:
            try:
                _ = np.load(img_path, mmap_mode="r")
                _ = np.load(lbl_path, mmap_mode="r")
            except Exception:
                needs_processing = True

        if needs_processing:
            process_args.append((tile_key, tile_groups[tile_key], config))

    with Pool(max(1, cpu_count() - 1)) as pool:
        for result in tqdm(pool.imap_unordered(process_tile, process_args),
                           total=len(process_args), desc="Processing"):
            if result:
                tile_key = result.split()[2]  # from "Processed tile <tile_key> (Zone XXX)"
                zone = tile_key.split("_")[0]
                print(f"[INFO] Zone {zone} → {result}")


if __name__ == "__main__":
    main()
