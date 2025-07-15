import os
import glob
import subprocess

# Folder containing prediction TIFFs
predictions_dir = "/media/lkm413/storage1/wetland_segmentation/outputs/predictions_Denmark2018/"
vrt_output_path = "/media/lkm413/storage1/wetland_segmentation/outputs/predictions_Denmark2018.vrt"

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
