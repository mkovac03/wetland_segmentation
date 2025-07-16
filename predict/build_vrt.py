import os
import glob
import subprocess
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
else:
    print("VRT successfully created.")
