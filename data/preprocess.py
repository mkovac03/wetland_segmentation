# File: data/preprocess_parallel.py
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

# ========== Remap definitions ==========
merge_map = {
    0: [0],
    1: [1],
    2: [2, 4, 6],
    3: [8],
    4: [9],
    5: [10],
    6: [12],
    7: [11, 13],
    8: [14],
    9: [15],
    10: [16, 17, 20, 21, 22],
    11: [18],
    12: [19]
}
remap_dict = {old: new for new, olds in merge_map.items() for old in olds}
label_names = {
    0: "No Wetland", 1: "Rice Fields", 2: "Riparian, fluvial and swamp forest",
    3: "Managed or grazed meadow", 4: "Wet grasslands", 5: "Wet heaths", 6: "Beaches", 7: "Inland marshes", 8: "Open mires",
    9: "Salt marshes", 10: "Surface water", 11: "Saltpans", 12: "Intertidal flats"
}
ignore_val = 255
nodata_val = -32768
num_classes = len(merge_map)

def process_file(args):
    f, config = args
    input_channels = config["input_channels"]
    target_crs = config["crs_target"]
    output_dir = config["processed_dir"]
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(f))[0].replace('_reproj', '')
    img_out = os.path.join(output_dir, f"{base}_img.npy")
    lbl_out = os.path.join(output_dir, f"{base}_lbl.npy")

    if os.path.exists(img_out) and os.path.exists(lbl_out):
        return None  # skip already processed tiles

    try:
        needs_crop = False
        with rasterio.open(f) as src:
            if src.crs is None or src.crs.to_string() != target_crs:
                dst_path = os.path.join(output_dir, os.path.basename(f).replace('.tif', '_reproj.tif'))
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': target_crs,
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
                            dst_crs=target_crs,
                            resampling=resampling
                        )
                out_path = dst_path
                needs_crop = True
            else:
                out_path = f

        with rasterio.open(out_path) as src:
            label = src.read(1).astype(np.int32)
            label[label == nodata_val] = ignore_val
            label = np.vectorize(lambda v: remap_dict.get(v, ignore_val))(label).astype(np.uint8)

            unexpected_ids = set(np.unique(label[label != ignore_val])) - set(range(num_classes))
            if unexpected_ids:
                return f"[ERROR] {f} unexpected labels: {unexpected_ids}"

            background_label = remap_dict.get(0)
            if background_label is not None:
                if (np.sum(label == background_label) / label.size) > 0.95:
                    return None  # skip silently

            if src.count < input_channels + 1:
                return None

            image = src.read(list(range(2, 2 + input_channels))).astype(np.float32)

            if needs_crop or image.shape[1:] != (512, 512):
                image = center_crop(image, (512, 512))
            if label.shape != (512, 512):
                label = center_crop(label, (512, 512))

            mask = label != ignore_val
            image = image * mask[None, :, :]

            base = os.path.splitext(os.path.basename(out_path))[0].replace('_reproj', '')
            np.save(os.path.join(output_dir, f"{base}_img.npy"), image)
            np.save(os.path.join(output_dir, f"{base}_lbl.npy"), label)

            return None  # silent success
    except Exception as e:
        return f"[FAIL] {f}: {str(e)}"

# ========== Main ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs("data", exist_ok=True)
    with open("data/label_remap.json", "w") as f:
        json.dump(remap_dict, f)
    with open("data/label_remap_longnames.json", "w") as f:
        json.dump(label_names, f, indent=2)

    files = glob.glob(os.path.join(config["input_dir"], "*.tif"))
    with Env(GDAL_NUM_THREADS="ALL_CPUS"):
        with Pool(max(cpu_count() - 1, 1)) as pool:
            for res in tqdm(pool.imap_unordered(process_file, [(f, config) for f in files]), total=len(files)):
                if res:
                    print(res)  # only errors are printed
