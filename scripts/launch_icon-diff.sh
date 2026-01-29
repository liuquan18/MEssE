#!/bin/bash
# Quick run script for diff_trainer.py

# Set paths
ICON_BUILD_DIR="/work/mh0033/m301250/20260126_ML-ICON/MEssE-2/build/messe_env/build_dir/icon-model"
COMIN_PLUGIN_SCRIPT="/work/mh0033/m301250/20260126_ML-ICON/MEssE-2/comin_plugin/diff_trainer.py"
LEVANTE_ACCOUNT="mh0033"

# Call the main run_icon.sh script
bash /work/mh0033/m301250/20260126_ML-ICON/MEssE-2/scripts/run_icon.sh \
    "$ICON_BUILD_DIR" \
    "$COMIN_PLUGIN_SCRIPT" \
    "$LEVANTE_ACCOUNT"
