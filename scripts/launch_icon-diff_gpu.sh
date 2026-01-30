#!/bin/bash
# GPU-enabled quick run script for diff_trainer_gpu.py
# Uses GPU partition with A100 GPU for 15-30× faster training

# Set paths
ICON_BUILD_DIR="/work/mh0033/m301250/20260126_ML-ICON/MEssE-2/build/messe_env/build_dir/icon-model"
COMIN_PLUGIN_SCRIPT="/work/mh0033/m301250/20260126_ML-ICON/MEssE-2/comin_plugin/diff_trainer_gpu.py"
LEVANTE_ACCOUNT="mh0033"

# Clean previous training data
echo "Cleaning previous training data..."
rm -rf /scratch/m/m301250/icon_exercise_comin/*
echo "Starting fresh training run"

# Configuration
START_DATE="1979-01-01T00:00:00Z"  # Start date (default: Jan 1, 1979)
END_DATE="1979-01-06T00:00:00Z"    # Run until Jan 6, 1979 (6 days total)
NUM_NODES=1                         # Use 1 node for GPU (GPU nodes have fewer CPUs)
TIME_LIMIT="3:00:00"               # Time limit: 3 hours (needed for 48-timestep buffer)

# Call the GPU-enabled run_icon_gpu.sh script
bash /work/mh0033/m301250/20260126_ML-ICON/MEssE-2/scripts/run_icon_gpu.sh \
    "$ICON_BUILD_DIR" \
    "$COMIN_PLUGIN_SCRIPT" \
    "$LEVANTE_ACCOUNT" \
    "$START_DATE" \
    "$END_DATE" \
    "$NUM_NODES" \
    "$TIME_LIMIT"
