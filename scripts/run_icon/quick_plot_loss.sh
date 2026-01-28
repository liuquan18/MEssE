#!/bin/bash
# Quick script to plot ComIn Mini-batch GNN loss curve

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work/mh1498/m301257/work/MEssE"

# Default parameters
LOG_DIR="/scratch/m/m301257/icon_exercise_comin"
OUTPUT_DIR="${BASE_DIR}/experiment"
VENV_PATH="${BASE_DIR}/environment/python/py_venv"

# Help message
usage() {
    cat << EOF
Usage: $0 [options] [job_id]

Quick plotting script for ComIn Mini-batch GNN training loss curves.
Generates 3-panel visualization: timestep avg (linear), timestep avg (log), and all batch losses.

Options:
  -h, --help          Show this help message
  -l, --log-dir DIR   Specify ComIn log directory (default: $LOG_DIR)
  -o, --output FILE   Specify output filename (default: loss_curve_gnn.png)
  -s, --slurm FILE    Use SLURM output file instead of log directory
  -t, --title TITLE   Custom chart title

Examples:
  $0                          # Use default log directory
  $0 22285966                 # Use SLURM file from GNN job 22285966
  $0 -o my_gnn_loss.png       # Custom output filename
  $0 -l /path/to/logs         # Specify log directory
  $0 -s experiment/slurm.22285966.out  # Direct SLURM file path

EOF
    exit 0
}

# Parse arguments
JOB_ID=""
OUTPUT_FILE="loss_curve_gnn.png"
SLURM_FILE=""
TITLE=""
USE_SLURM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -s|--slurm)
            SLURM_FILE="$2"
            USE_SLURM=true
            shift 2
            ;;
        -t|--title)
            TITLE="$2"
            shift 2
            ;;
        [0-9]*)
            JOB_ID="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Activate virtual environment
echo "Activating Python virtual environment..."
source "${VENV_PATH}/bin/activate"

# Build command
CMD="python ${SCRIPT_DIR}/plot_comin_loss.py"

# If job ID specified, use corresponding SLURM file
if [ -n "$JOB_ID" ]; then
    SLURM_FILE="${OUTPUT_DIR}/slurm.${JOB_ID}.out"
    if [ ! -f "$SLURM_FILE" ]; then
        echo "Error: SLURM file does not exist: $SLURM_FILE"
        exit 1
    fi
    USE_SLURM=true
    echo "Using SLURM file: $SLURM_FILE"
fi

# Add arguments
if [ "$USE_SLURM" = true ]; then
    CMD="$CMD --slurm-file $SLURM_FILE"
else
    CMD="$CMD --log-dir $LOG_DIR"
fi

CMD="$CMD --output ${OUTPUT_DIR}/${OUTPUT_FILE}"

if [ -n "$TITLE" ]; then
    CMD="$CMD --title \"$TITLE\""
fi

# Display command
echo "Executing command:"
echo "$CMD"
echo ""

# Execute plotting
eval $CMD

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Plotting successful!"
    echo "  Image location: ${OUTPUT_DIR}/${OUTPUT_FILE}"
    
    # Provide download hint if on compute node
    if [[ $(hostname) == l* ]]; then
        echo ""
        echo "Hint: To view the image, run from your local computer:"
        echo "  scp levante:${OUTPUT_DIR}/${OUTPUT_FILE} ."
    fi
else
    echo ""
    echo "✗ Plotting failed, please check error messages above"
    exit 1
fi
