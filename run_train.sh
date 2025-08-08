#!/bin/bash

# Move to the script's directory
cd "$(dirname "$0")"

# Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wetland_segmentation

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

# Get timestamp
if [[ -n "$1" ]]; then
  NOW="$1"
else
  NOW=$(date +%Y%m%d_%H%M%S)
fi
echo "[INFO] Using timestamp: $NOW"

# Expand config
CONFIG_SRC="configs/config.yaml"
CONFIG_EXPANDED="configs/config_expanded.yaml"
sed "s|{now}|$NOW|g" "$CONFIG_SRC" > "$CONFIG_EXPANDED"
#
### Run preprocessing
#python data/preprocess.py --config "$CONFIG_EXPANDED"
#
## Run split_data.py to generate splits and weights
#python split_data.py --config "$CONFIG_EXPANDED"

# Kill stale processes
echo "[INFO] Cleaning up previous runs..."
pkill -f -u "$USER" "train/train.py" || true

# Run training
python train/train.py --config "$CONFIG_EXPANDED"
