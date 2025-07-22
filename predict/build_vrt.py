import os
import glob
import subprocess
import json
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from datetime import datetime
import yaml

# Load target CRS from config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

target_crs = config["crs_target"]
base_dir = "/media/lkm413/storage1/wetland_segmentation/outputs/"

# Find latest predictions folder
pattern = os.path.join(base_dir, "predictions_Denmark2018_*")
prediction_folders = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
if not prediction_folders:
    raise RuntimeError("No timestamped prediction folders found.")

predictions_dir = prediction_folders[0]
timestamp = os.path.basename(predictions_dir).split("_")[-1]
vrt_output_path = os.path.join(base_dir, f"predictions_Denmark2018_{timestamp}.vrt")
qml_output_path = os.path.join(base_dir, f"predictions_Denmark2018_{timestamp}.qml")

# Prepare reprojection directory
reproj_dir = os.path.join(predictions_dir, "reprojected")
os.makedirs(reproj_dir, exist_ok=True)

# Collect and validate all TIFs
tif_files = glob.glob(os.path.join(predictions_dir, "*.tif"))
if len(tif_files) == 0:
    raise RuntimeError(f"No TIFF files found in {predictions_dir}")

print(f"[INFO] Checking CRS for {len(tif_files)} tiles...")

reprojected_files = []
for tif in tif_files:
    with rasterio.open(tif) as src:
        src_crs = src.crs.to_string()
        if src_crs != target_crs:
            # Reproject this file
            dst_path = os.path.join(reproj_dir, os.path.basename(tif))
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height
            })
            with rasterio.open(dst_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest if i == 1 else Resampling.bilinear
                    )
            print(f"[INFO] Reprojected: {os.path.basename(tif)}")
            reprojected_files.append(dst_path)
        else:
            reprojected_files.append(tif)

# Build VRT
cmd = ["gdalbuildvrt", "-overwrite", vrt_output_path] + reprojected_files
print(f"[INFO] Building VRT at {vrt_output_path} with {len(reprojected_files)} tiles...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print("Error building VRT:")
    print(result.stderr)
    exit(1)
else:
    print("VRT successfully created.")

# Load label names
try:
    with open("data/label_remap_longnames.json", "r") as f:
        label_names = json.load(f)
except FileNotFoundError:
    print("[WARNING] label_remap_longnames.json not found. Cannot create QML.")
    exit(0)

# Generate QML with class 0 transparent
color_table = [
    "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928", "#a6cee3",
    "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99", "#8dd3c7", "#ffffb3"
]

qml_lines = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<qgis>',
    '  <renderer-v2 type="paletted" forcerasterrenderer="0" band="1">',
    '    <paletteEntries>'
]

for i, (class_id, name) in enumerate(sorted((int(k), v) for k, v in label_names.items())):
    color = color_table[i % len(color_table)].lstrip("#")
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    alpha = 0 if class_id == 0 else 255
    qml_lines.append(
        f'      <paletteEntry value="{class_id}" label="{name}" color="{r},{g},{b},{alpha}"/>'
    )

qml_lines += [
    '    </paletteEntries>',
    '  </renderer-v2>',
    '</qgis>'
]

with open(qml_output_path, "w") as f:
    f.write("\n".join(qml_lines))

print(f"[INFO] QML file written to {qml_output_path}")
