import os
import glob
import re
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ========= File Paths ==========
grid_path = "/media/lkm413/storage21/gee_embedding_download/eu_grid_5120m_epsg3035.gpkg"
dk_path = "/media/lkm413/storage4/country_shp/geometry_DK.shp"

input_dirs = [
    "/media/lkm413/storage21/gee_embedding_download/images/Europe/2018/bands_01_22",
    "/media/lkm413/storage21/gee_embedding_download/images/Europe/2018/bands_23_44",
    "/media/lkm413/storage21/gee_embedding_download/images/Europe/2018/bands_45_63"
]

output_dir = "/media/lkm413/storage3/DK_PEAT/images/embeddings"
os.makedirs(output_dir, exist_ok=True)

nodata_val = -32768
remove_indices = {0, 23, 46}  # global band indices to remove

# ========= Load and Intersect Grid ==========
print("[INFO] Loading grid and Denmark boundary...")
grid = gpd.read_file(grid_path)
dk = gpd.read_file(dk_path).to_crs(grid.crs)

print("[INFO] Finding intersecting tiles...")
intersecting = gpd.sjoin(grid, dk, predicate='intersects')
intersecting_ids = sorted(intersecting.index.values - 1)

print(f"[INFO] Intersecting tile indices: {len(intersecting_ids)}")

# ========= Build tile mapping ==========
def extract_tile_index(filename):
    match = re.search(r'_(\d+)\.tif$', filename)
    return match.group(1) if match else None

tile_files = {}
for dir_path in input_dirs:
    for path in glob.glob(os.path.join(dir_path, "*_3263[23]_*.tif")):
        tile_idx = extract_tile_index(path)
        if tile_idx and int(tile_idx) in intersecting_ids:
            tile_files.setdefault(tile_idx, []).append(path)

print(f"[INFO] Found {len(tile_files)} intersecting tile groups")

# ========= Clean Output Dir ==========
existing_outputs = glob.glob(os.path.join(output_dir, "tile_*.tif"))
for out in existing_outputs:
    idx = extract_tile_index(out)
    if idx and int(idx) not in intersecting_ids:
        os.remove(out)
        print(f"[INFO] Deleted non-DK tile: {out}")

# ========= Tile Stacking ==========
def stack_tile_bands(paths):
    bands = []
    meta = None
    global_band_idx = 0

    for path in sorted(paths):
        with rasterio.open(path) as src:
            if meta is None:
                meta = src.meta.copy()
                meta.update({
                    "count": 0,
                    "driver": "GTiff",
                    "nodata": src.nodata if src.nodata is not None else nodata_val
                })

            for i in range(1, src.count + 1):
                if global_band_idx in remove_indices:
                    global_band_idx += 1
                    continue
                band = src.read(i)
                bands.append(band)
                global_band_idx += 1

    stacked = np.stack(bands, axis=0)
    meta["count"] = stacked.shape[0]
    return stacked, meta

# ========= Processing Function ==========
def process_tile(tile_info):
    tile_idx, paths = tile_info
    out_path = os.path.join(output_dir, f"tile_{tile_idx}.tif")
    if os.path.exists(out_path):
        return None
    try:
        stacked, meta = stack_tile_bands(paths)
        with rasterio.open(out_path, "w", **meta) as dst:
            for i in range(stacked.shape[0]):
                dst.write(stacked[i], i + 1)
        return None
    except Exception as e:
        return f"[ERROR] Tile {tile_idx}: {e}"

# ========= Run Parallel ==========
tile_items = sorted(tile_files.items())
with Pool(cpu_count() - 1) as pool:
    for result in tqdm(pool.imap_unordered(process_tile, tile_items), total=len(tile_items)):
        if result:
            print(result)
