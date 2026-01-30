#!/bin/bash
# Quick run script for diff_trainer.py

# Set paths
ICON_BUILD_DIR="/work/mh1498/m301257/project2/MEssE/build/messe_env/build_dir/icon-model"
COMIN_PLUGIN_SCRIPT="/work/mh1498/m301257/project2/MEssE/comin_plugin/diff_trainer.py"
LEVANTE_ACCOUNT="mh1498"

# Configuration
START_DATE="1979-01-01T00:00:00Z"  # Start date (default: Jan 1, 1979)
END_DATE="1979-03-31T00:00:00Z"    # Run until Mar 31, 1979 (90 days)
NUM_NODES=1                         # Use 1 node

# Call the main run_icon.sh script with additional parameters
bash /work/mh1498/m301257/project2/MEssE/scripts/run_icon.sh \
    "$ICON_BUILD_DIR" \
    "$COMIN_PLUGIN_SCRIPT" \
    "$LEVANTE_ACCOUNT" \
    "$START_DATE" \
    "$END_DATE" \
    "$NUM_NODES"
