import os
import glob
import subprocess
import json
from datetime import datetime

# Base predictions directory
base_dir = "/media/lkm413/storage1/wetland_segmentation/outputs/"

# Find latest predictions_Denmark2018_* folder
pattern = os.path.join(base_dir, "predictions_Denmark2018_*")
prediction_folders = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

if not prediction_folders:
    raise RuntimeError("No timestamped prediction folders found.")

predictions_dir = prediction_folders[0]
timestamp = os.path.basename(predictions_dir).split("_")[-1]
vrt_output_path = os.path.join(base_dir, f"predictions_Denmark2018_{timestamp}.vrt")
qml_output_path = os.path.join(base_dir, f"predictions_Denmark2018_{timestamp}.qml")

# Get all .tif files
tif_files = glob.glob(os.path.join(predictions_dir, "*.tif"))
if len(tif_files) == 0:
    raise RuntimeError(f"No TIFF files found in {predictions_dir}")

# Build VRT command
cmd = ["gdalbuildvrt", "-overwrite", vrt_output_path] + tif_files
print(f"Building VRT at {vrt_output_path} with {len(tif_files)} tiles...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print("Error building VRT:")
    print(result.stderr)
    exit(1)
else:
    print("VRT successfully created.")

# Load label names from remap file
try:
    with open("data/label_remap_longnames.json", "r") as f:
        label_names = json.load(f)
except FileNotFoundError:
    print("[WARNING] label_remap_longnames.json not found. Cannot create QML.")
    exit(0)

# Generate a discrete QML with transparency for No Wetland (ID 0)
color_table = [
    "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928", "#a6cee3",
    "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99", "#8dd3c7", "#ffffb3"
]  # At least 14 distinct colors

qml_lines = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<qgis>',
    '  <renderer-v2 type="paletted" forcerasterrenderer="0" band="1">',
    '    <paletteEntries>'
]

# Sort by integer class ID
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

print(f"QML file written to {qml_output_path}")
