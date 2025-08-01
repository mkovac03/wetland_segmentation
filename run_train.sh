#!/bin/bash

cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wetland_segmentation
export PYTHONPATH=$(pwd)

# Timestamp setup
if [[ -n "$1" ]]; then
  NOW="$1"
else
  NOW=$(date +%Y%m%d_%H%M%S)
fi
echo "Using timestamp: $NOW"

# Config prep
CONFIG_SRC="configs/config.yaml"
CONFIG_EXPANDED="configs/config_expanded.yaml"
sed "s|{now}|$NOW|g" "$CONFIG_SRC" > "$CONFIG_EXPANDED"

# Parse paths from config
INPUT_DIR=$(grep '^input_dir:' "$CONFIG_EXPANDED" | awk '{print $2}' | tr -d '"')
PROCESSED_DIR=$(grep '^processed_dir:' "$CONFIG_EXPANDED" | awk '{print $2}' | tr -d '"')

# Count all expected .tif input tiles
EXPECTED=$(find "$INPUT_DIR" -name '*.tif' | wc -l)

# Count valid _img.npy + _lbl.npy pairs
FOUND=$(find "$PROCESSED_DIR" -name '*_img.npy' | sed 's/_img.npy//' | while read base; do
  lbl="${base}_lbl.npy"
  [[ -f "$lbl" ]] && echo "$base"
done | wc -l)

if [[ "$FOUND" -ge "$EXPECTED" ]]; then
  echo "✅ Preprocessing already complete for $NOW ($FOUND / $EXPECTED valid tiles). Skipping..."
else
  echo "🚧 Running preprocessing ($FOUND / $EXPECTED valid tiles done)..."
  python data/preprocess.py --config "$CONFIG_EXPANDED"
fi

# Check if splits exist
SPLIT_FILE="data/splits/splits_${NOW}.json"
if [[ -f "$SPLIT_FILE" ]]; then
  echo "✅ Splits file already exists: $SPLIT_FILE. Skipping..."
else
  echo "📂 Generating splits..."
  python data/split_data.py --config "$CONFIG_EXPANDED"
fi

# Start training
echo "🚀 Starting training..."
PYTHONPATH=$(pwd) python train/train.py --config "$CONFIG_EXPANDED" &

TRAIN_PID=$!
sleep 5

# Start TensorBoard if not already running
TENSORBOARD_PORT=6006
if ! lsof -i:$TENSORBOARD_PORT >/dev/null; then
  mkdir -p outputs
  nohup tensorboard --logdir=outputs/ --port=$TENSORBOARD_PORT > outputs/tensorboard.log 2>&1 &
  echo "📊 TensorBoard started at http://localhost:$TENSORBOARD_PORT"
else
  echo "📊 TensorBoard already running on port $TENSORBOARD_PORT"
fi

wait $TRAIN_PID
