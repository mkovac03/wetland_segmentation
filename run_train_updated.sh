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

CONFIG_SRC="configs/config.yaml"
ZONES=($(find data/processed -name '*_img.npy' | sed -n 's/.*_\(326[0-9][0-9]\)_.*_img.npy/\1/p' | sort -u))

PREV_CKPT=""

for ZONE in "${ZONES[@]}"; do
  echo "=========================="
  echo "🌍 Training zone EPSG:$ZONE"
  echo "=========================="

  ZID="epsg${ZONE}_${NOW}"
  CONFIG_ZONE="configs/config_${ZID}.yaml"
  PROCESSED_DIR="data/processed/${ZID}"
  SPLITS_PATH="data/splits/splits_${ZID}.json"
  OUTPUT_DIR="outputs/${ZID}"

  mkdir -p "$PROCESSED_DIR"

  sed "s|{now}|${ZID}|g" "$CONFIG_SRC" > "$CONFIG_ZONE"
  sed -i "s|splits_path:.*|splits_path: $SPLITS_PATH|" "$CONFIG_ZONE"
  sed -i "s|processed_dir:.*|processed_dir: $PROCESSED_DIR|" "$CONFIG_ZONE"
  sed -i "s|output_dir:.*|output_dir: $OUTPUT_DIR|" "$CONFIG_ZONE"

  echo "📂 Preprocessing for $ZONE..."
  python data/preprocess.py --config "$CONFIG_ZONE"

  echo "🧩 Generating splits for $ZONE..."
  python split_data.py --config "$CONFIG_ZONE"

  echo "🚀 Training model for $ZONE..."
  if [[ -n "$PREV_CKPT" && -f "$PREV_CKPT" ]]; then
    python train/train.py --config "$CONFIG_ZONE" --resume "$PREV_CKPT"
  else
    python train/train.py --config "$CONFIG_ZONE"
  fi

  PREV_CKPT="$OUTPUT_DIR/best_model.pt"

  echo "✅ Finished $ZONE → best model: $PREV_CKPT"
done

# TensorBoard section (optional)
TENSORBOARD_PORT=6006
if ! lsof -i:$TENSORBOARD_PORT >/dev/null; then
  mkdir -p outputs
  nohup tensorboard --logdir=outputs/ --port=$TENSORBOARD_PORT > outputs/tensorboard.log 2>&1 &
  echo "📊 TensorBoard started at http://localhost:$TENSORBOARD_PORT"
else
  echo "📊 TensorBoard already running on port $TENSORBOARD_PORT"
fi