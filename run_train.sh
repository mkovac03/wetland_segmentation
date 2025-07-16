#!/bin/bash

cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wetland_segmentation
export PYTHONPATH=$(pwd)

if [[ -n "$1" ]]; then
  NOW="$1"
else
  NOW=$(date +%Y%m%d_%H%M%S)
fi
echo "Using timestamp: $NOW"

CONFIG_SRC="configs/config.yaml"
CONFIG_EXPANDED="configs/config_expanded.yaml"
sed "s|{now}|$NOW|g" "$CONFIG_SRC" > "$CONFIG_EXPANDED"

# Check if preprocessing is already done
PROCESSED_DIR="data/processed/$NOW"
EXPECTED_COUNT=2186  # update if your tile count changes
ACTUAL_COUNT=$(find "$PROCESSED_DIR" -type f \( -name "*.npy" -o -name "*.tif" \) 2>/dev/null | wc -l)

if [[ "$ACTUAL_COUNT" -ge "$EXPECTED_COUNT" ]]; then
  echo "Preprocessing already done for $NOW ($ACTUAL_COUNT files). Skipping..."
else
  echo "Running preprocessing..."
  python data/preprocess.py --config "$CONFIG_EXPANDED"
fi

# Check if splits already exist
SPLIT_FILE="data/splits/splits_${NOW}.json"
if [[ -f "$SPLIT_FILE" ]]; then
  echo "Splits file already exists: $SPLIT_FILE. Skipping..."
else
  echo "Generating splits..."
  python data/split_data.py --config "$CONFIG_EXPANDED"
fi

echo "Starting training..."
PYTHONPATH=$(pwd) python train/train.py --config "$CONFIG_EXPANDED" &

TRAIN_PID=$!
sleep 5
mkdir -p outputs
nohup tensorboard --logdir=outputs/ --port=6006 > outputs/tensorboard.log 2>&1 &
echo "TensorBoard started at http://localhost:6006"

wait $TRAIN_PID
