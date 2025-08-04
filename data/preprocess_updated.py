# Updated preprocess.py with 64-band merging logic across band chunks + reprojection to EPSG:3035

import os
import glob
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.env import Env
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import json
from multiprocessing import Pool, cpu_count

# ========== Remap definitions ==========
merge_map = {
    0: [0], 1: [1], 2: [2, 4, 6], 3: [8], 4: [9], 5: [10], 6: [12],
    7: [11, 13], 8: [14], 9: [15], 10: [16, 17, 20, 21, 22],
    11: [18], 12: [19]
}
remap_dict = {old: new for new, olds in merge_map.items() for old in olds}
label_names = {
    0: "No Wetland", 1: "Rice Fields", 2: "Riparian, fluvial and swamp forest",
    3: "Managed or grazed meadow", 4: "Wet grasslands", 5: "Wet heaths", 6: "Beaches",
    7: "Inland marshes", 8: "Open mires", 9: "Salt marshes", 10: "Surface water",
    11: "Saltpans", 12: "Intertidal flats"
}
ignore_val = 255
nodata_val = -32768
num_classes = len(merge_map)

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
        raise ValueError(f"Unsupported array shape for cropping: {array.shape}")

def is_valid_npy(path):
    try:
        arr = np.load(path)
        return arr.size > 0
    except:
        return False

def match_tile_triplets(input_dirs):
    basename_groups = []
    for d in input_dirs:
        tifs = glob.glob(os.path.join(d, "*.tif"))
        names = {os.path.basename(f).split("_")[-1][:-4]: f for f in tifs}
        basename_groups.append(names)

    common_keys = set.intersection(*[set(g.keys()) for g in basename_groups])
    return [(basename_groups[0][k], basename_groups[1][k], basename_groups[2][k]) for k in sorted(common_keys)]

def reproject_array(src, band_indices, dst_crs):
    arrays = []
    for i in band_indices:
        band = src.read(i)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        dest = np.empty((height, width), dtype=band.dtype)
        resampling = Resampling.nearest if i == 1 else Resampling.bilinear
        reproject(
            source=band,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=resampling
        )
        arrays.append(dest)
    return np.stack(arrays)

def process_file(args):
    f1, f2, f3, config = args
    input_channels = config["input_channels"]
    target_crs = config["crs_target"]
    output_dir = config["processed_dir"]

    base = os.path.splitext(os.path.basename(f1))[0].replace("bands_01_22_", "")
    img_out = os.path.join(output_dir, f"{base}_img.npy")
    lbl_out = os.path.join(output_dir, f"{base}_lbl.npy")

    if os.path.exists(img_out) and os.path.exists(lbl_out):
        if is_valid_npy(img_out) and is_valid_npy(lbl_out):
            return None

    try:
        with Env(GDAL_NUM_THREADS="ALL_CPUS"):
            with rasterio.open(f1) as src1, rasterio.open(f2) as src2, rasterio.open(f3) as src3:
                label = reproject_array(src1, [1], target_crs)[0].astype(np.int32)
                label[label == nodata_val] = ignore_val
                label = np.vectorize(lambda v: remap_dict.get(v, ignore_val))(label).astype(np.uint8)

                unexpected_ids = set(np.unique(label[label != ignore_val])) - set(range(num_classes))
                if unexpected_ids:
                    return f"[ERROR] {base} unexpected labels: {unexpected_ids}"

                background_label = remap_dict.get(0)
                if background_label is not None:
                    if (np.sum(label == background_label) / label.size) > 0.95:
                        np.save(img_out, np.zeros((input_channels, 512, 512), dtype=np.float32))
                        np.save(lbl_out, np.full((512, 512), ignore_val, dtype=np.uint8))
                        return None

                c0 = reproject_array(src1, list(range(2, src1.count + 1)), target_crs)
                c1 = reproject_array(src2, list(range(1, src2.count + 1)), target_crs)
                c2 = reproject_array(src3, list(range(1, src3.count + 1)), target_crs)
                image = np.concatenate([c0, c1, c2], axis=0).astype(np.float32)

                if image.shape[1:] != (512, 512):
                    image = center_crop(image, (512, 512))
                if label.shape != (512, 512):
                    label = center_crop(label, (512, 512))

                mask = label != ignore_val
                image = image * mask[None, :, :]
                image[:, ~mask] = 0

                np.save(img_out, image)
                np.save(lbl_out, label)
                return None

    except Exception as e:
        return f"[FAIL] {base}: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["processed_dir"], exist_ok=True)
    input_dirs = config["input_dirs"]

    tile_triplets = match_tile_triplets(input_dirs)
    print(f"[INFO] Found {len(tile_triplets)} valid tile triplets across all band folders.")

    with Pool(min(cpu_count() // 2, 4)) as pool:
        for res in tqdm(pool.imap_unordered(lambda a: process_file((*a, config)), tile_triplets),
                        total=len(tile_triplets)):
            if res:
                print(res)
