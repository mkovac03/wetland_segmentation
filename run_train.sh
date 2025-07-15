#!/bin/bash

# Ensure we're in the project root
cd "$(dirname "$0")"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wetland_segmentation

# Add project root to PYTHONPATH
export PYTHONPATH=$(pwd)

# Set a consistent timestamp
NOW=$(date +%Y%m%d_%H%M%S)
echo "Using timestamp: $NOW"

# Replace {now} in config and write to a temp expanded config
CONFIG_SRC="configs/config.yaml"
CONFIG_EXPANDED="configs/config_expanded.yaml"
sed "s|{now}|$NOW|g" $CONFIG_SRC > $CONFIG_EXPANDED

echo "Running preprocessing..."
python data/preprocess.py --config $CONFIG_EXPANDED

echo "Generating splits..."
python data/split_data.py --config $CONFIG_EXPANDED

echo "Starting training..."
PYTHONPATH=$(pwd) python train/train.py --config $CONFIG_EXPANDED &

TRAIN_PID=$!

sleep 5
mkdir -p outputs
nohup tensorboard --logdir=outputs/ --port=6006 > outputs/tensorboard.log 2>&1 &
echo "TensorBoard started at http://localhost:6006"

wait $TRAIN_PID
