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


def is_valid_npy(path):
    try:
        arr = np.load(path)
        return arr.size > 0
    except:
        return False


def process_file(args):
    f, config = args
    input_channels = config["input_channels"]
    target_crs = config["crs_target"]
    output_dir = config["processed_dir"]

    base = os.path.splitext(os.path.basename(f))[0]
    img_out = os.path.join(output_dir, f"{base}_img.npy")
    lbl_out = os.path.join(output_dir, f"{base}_lbl.npy")

    if os.path.exists(img_out) and os.path.exists(lbl_out):
        if is_valid_npy(img_out) and is_valid_npy(lbl_out):
            return None

    try:
        with Env(GDAL_NUM_THREADS="ALL_CPUS"):
            with rasterio.open(f) as src:
                if src.crs is None:
                    return f"[ERROR] {f} has no CRS"
                if src.count < input_channels + 1:
                    return f"[ERROR] {f} has only {src.count} bands but {input_channels + 1} needed"

                if src.crs.to_string() != target_crs:
                    transform, width, height = calculate_default_transform(
                        src.crs, target_crs, src.width, src.height, *src.bounds)
                    reproj_arrays = []
                    for i in range(1, src.count + 1):
                        dest = np.empty((height, width), dtype=src.dtypes[i - 1])
                        resampling = Resampling.nearest if i == 1 else Resampling.bilinear
                        reproject(
                            source=rasterio.band(src, i),
                            destination=dest,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=resampling
                        )
                        reproj_arrays.append(dest)
                    needs_crop = True
                else:
                    reproj_arrays = [src.read(i) for i in range(1, src.count + 1)]
                    needs_crop = False

                label = reproj_arrays[0].astype(np.int32)
                label[label == nodata_val] = ignore_val
                label = np.vectorize(lambda v: remap_dict.get(v, ignore_val))(label).astype(np.uint8)

                unexpected_ids = set(np.unique(label[label != ignore_val])) - set(range(num_classes))
                if unexpected_ids:
                    return f"[ERROR] {f} unexpected labels: {unexpected_ids}"

                background_label = remap_dict.get(0)
                if background_label is not None:
                    if (np.sum(label == background_label) / label.size) > 0.95:
                        return None

                image = np.stack(reproj_arrays[1:1 + input_channels]).astype(np.float32)

                if needs_crop or image.shape[1:] != (512, 512):
                    image = center_crop(image, (512, 512))
                if label.shape != (512, 512):
                    label = center_crop(label, (512, 512))

                mask = label != ignore_val
                image = image * mask[None, :, :]

                np.save(img_out, image)
                np.save(lbl_out, label)

                return None

    except Exception as e:
        return f"[FAIL] {f}: {str(e)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["processed_dir"], exist_ok=True)
    input_files = sorted(glob.glob(os.path.join(config["input_dir"], "*.tif")))

    processed_img_files = glob.glob(os.path.join(config["processed_dir"], "*_img.npy"))
    resume_index = len(processed_img_files)

    if resume_index >= len(input_files):
        print("[INFO] All files already processed. Skipping preprocessing.")
        input_files = []
    else:
        print(f"[INFO] Resuming from index {resume_index}/{len(input_files)}")
        input_files = input_files[resume_index:]

    if input_files:
        with Pool(max(cpu_count() - 1, 1)) as pool:
            for res in tqdm(pool.imap_unordered(process_file, [(f, config) for f in input_files]), total=len(input_files)):
                if res:
                    print(res)
