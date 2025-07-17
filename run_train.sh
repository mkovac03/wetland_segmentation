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

# Check if preprocessing already done
PROCESSED_DIR="data/processed/$NOW"
if compgen -G "$PROCESSED_DIR/*.npy" > /dev/null; then
  echo "Preprocessing already done for $NOW ($(find "$PROCESSED_DIR" -type f -name '*.npy' | wc -l) files). Skipping..."
else
  echo "Running preprocessing..."
  python data/preprocess.py --config "$CONFIG_EXPANDED"
fi


# Check if splits exist
SPLIT_FILE="data/splits/splits_${NOW}.json"
if [[ -f "$SPLIT_FILE" ]]; then
  echo "Splits file already exists: $SPLIT_FILE. Skipping..."
else
  echo "Generating splits..."
  python data/split_data.py --config "$CONFIG_EXPANDED"
fi

# Start training
echo "Starting training..."
PYTHONPATH=$(pwd) python train/train.py --config "$CONFIG_EXPANDED" &

TRAIN_PID=$!
sleep 5

# Start TensorBoard if not already running
TENSORBOARD_PORT=6006
if ! lsof -i:$TENSORBOARD_PORT >/dev/null; then
  mkdir -p outputs
  nohup tensorboard --logdir=outputs/ --port=$TENSORBOARD_PORT > outputs/tensorboard.log 2>&1 &
  echo "TensorBoard started at http://localhost:$TENSORBOARD_PORT"
else
  echo "TensorBoard already running on port $TENSORBOARD_PORT"
fi

wait $TRAIN_PID
