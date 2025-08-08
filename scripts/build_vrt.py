import os
import subprocess
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ========== Config ==========
input_dir = "/media/lkm413/storage3/DK_PEAT/images/embeddings"
vrt_dir = "/media/lkm413/storage3/DK_PEAT/vrt"
reproj_dir = os.path.join(vrt_dir, "reprojected")
target_crs = "EPSG:32632"

os.makedirs(vrt_dir, exist_ok=True)
os.makedirs(reproj_dir, exist_ok=True)

tile_list_txt = os.path.join(vrt_dir, "tile_list_epsg32632.txt")
vrt_output = os.path.join(vrt_dir, "embeddings_band1_epsg32632.vrt")

# ========== Collect all tiles ==========
tile_paths = sorted([
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if f.startswith("tile_") and f.endswith(".tif")
])

print(f"[INFO] Found {len(tile_paths)} tile files.")
print(f"[INFO] Target projection: {target_crs}")

valid_tiles = []
to_reproject = []

# ========== First Pass: Check CRS ==========
for path in tile_paths:
    with rasterio.open(path) as src:
        if src.crs.to_string() == target_crs:
            valid_tiles.append(path)
        else:
            dst_path = os.path.join(reproj_dir, os.path.basename(path))
            if os.path.exists(dst_path):
                valid_tiles.append(dst_path)
            else:
                valid_tiles.append(dst_path)
                to_reproject.append((path, dst_path))

print(f"[INFO] Tiles needing reprojection: {len(to_reproject)}")

# ========== Reprojection Function ==========
def reproject_tile(args):
    src_path, dst_path = args
    try:
        with rasterio.open(src_path) as src:
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
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )
        return None
    except Exception as e:
        return f"[ERROR] {src_path} → {e}"

# ========== Run Reprojections in Parallel ==========
if to_reproject:
    print("[INFO] Starting parallel reprojection...")
    with Pool(cpu_count() - 1) as pool:
        for result in tqdm(pool.imap_unordered(reproject_tile, to_reproject), total=len(to_reproject)):
            if result:
                print(result)

# ========== Write Final List ==========
with open(tile_list_txt, "w") as f:
    for path in valid_tiles:
        f.write(path + "\n")

print(f"[INFO] Wrote {len(valid_tiles)} aligned tile paths to {tile_list_txt}")

# ========== Build VRT ==========
cmd = [
    "gdalbuildvrt",
    "-b", "1",
    "-input_file_list", tile_list_txt,
    vrt_output
]

print(f"[INFO] Building VRT at {vrt_output}...")
subprocess.run(cmd, check=True)
print("[✅] VRT build complete.")
