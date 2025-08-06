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

# ========== Tile Processing ==========

def compute_histogram(tile_key, tif_list, config):
    try:
        target_crs = config["crs_target"]
        processed_dir = config["processed_dir"]
        remap_dict = config["label_remap"]
        ignore_val = config["ignore_val"]
        nodata_val = config["nodata_val"]

        tif_list = sorted(tif_list)
        bands = []

        for path in tif_list:
            with rasterio.open(path) as src:
                array, _ = reproject_stack(src, target_crs)
                bands.extend(array)

        image = np.stack(bands)
        label = image[0].astype(np.int32)
        label[label == nodata_val] = ignore_val
        remapped = np.vectorize(lambda v: remap_dict.get(v, ignore_val))(label).astype(np.uint8)

        hist_dir = os.path.join(processed_dir, "label_histograms")
        os.makedirs(hist_dir, exist_ok=True)
        classes, counts = np.unique(remapped[remapped != ignore_val], return_counts=True)
        hist = dict(zip(map(str, classes.tolist()), counts.tolist()))

        with open(os.path.join(hist_dir, f"tile_{tile_key}.json"), "w") as f:
            json.dump({"tile_id": tile_key, "histogram": hist}, f)

        return tile_key, hist, remapped.shape[0], remapped.shape[1]

    except Exception as e:
        return tile_key, {"error": str(e)}, 0, 0

def should_process(hist, shape, threshold):
    if "error" in hist:
        return False
    total = shape[0] * shape[1]
    background = hist.get("0", 0)
    return (background / total) < threshold

def label_entropy(hist):
    values = np.array(list(hist.values()))
    prob = values / values.sum()
    return entropy(prob)

def process_tile(tile_key, tif_list, config):
    try:
        target_crs = config["crs_target"]
        patch_size = config.get("patch_size", 512)
        processed_dir = config["processed_dir"]
        remap_dict = config["label_remap"]
        ignore_val = config["ignore_val"]
        nodata_val = config["nodata_val"]

        out_path = os.path.join(processed_dir, f"tile_{tile_key}.tif")

        tif_list = sorted(tif_list)
        bands = []
        transforms = []

        for path in tif_list:
            with rasterio.open(path) as src:
                array, transform = reproject_stack(src, target_crs)
                bands.extend(array)
                transforms.append(transform)

        image = np.stack(bands)

        if image.shape[0] >= 3:
            total_pixels = image.shape[1] * image.shape[2]
            b2_ratio = np.sum(image[1] == nodata_val) / total_pixels
            b3_ratio = np.sum(image[2] == nodata_val) / total_pixels
            if b2_ratio > 0.10 and b3_ratio > 0.10:
                return None

        label = image[0].astype(np.int32)
        label[label == nodata_val] = ignore_val
        image[0] = np.vectorize(lambda v: remap_dict.get(v, ignore_val))(label).astype(np.uint8)

        if not np.any(image[0] != ignore_val):
            return None

        image = center_crop(image, target_size=(patch_size, patch_size))

        meta = {
            "driver": "GTiff",
            "height": patch_size,
            "width": patch_size,
            "count": image.shape[0],
            "dtype": image.dtype,
            "crs": target_crs,
            "transform": transforms[0],
            "compress": "lzw",
            "BIGTIFF": "IF_SAFER"
        }

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(image)

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

    args_list = [(tile_key, tif_list, config) for tile_key, tif_list in tile_groups.items()]
    with Pool(max(1, cpu_count() - 1)) as pool:
        results = list(tqdm(pool.imap_unordered(lambda args: compute_histogram(*args), args_list), total=len(args_list)))

    valid_tiles = [(tile_key, hist) for tile_key, hist, h, w in results if should_process(hist, (h, w), bg_thresh)]
    print(f"[INFO] {len(valid_tiles)} tiles pass background threshold < {bg_thresh}")

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

    process_args = [(tile_key, tile_groups[tile_key], config) for tile_key, _ in selected]
    with Pool(max(1, cpu_count() - 1)) as pool:
        for result in tqdm(pool.imap_unordered(lambda args: process_tile(*args), process_args), total=len(process_args), desc="Processing"):
            if result:
                print(result)

if __name__ == "__main__":
    main()
