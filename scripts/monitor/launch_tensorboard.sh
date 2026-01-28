#!/bin/bash
#
# TensorBoard Launcher for ICON Training Monitoring
# This script launches TensorBoard to visualize training metrics in real-time
#

# Get username
USER=$(whoami)

# Set log directory path
LOG_DIR="/scratch/${USER:0:1}/${USER}/icon_exercise_comin/runs"

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory does not exist: $LOG_DIR"
    echo "Please run the ICON model with the plugin first to generate training logs."
    exit 1
fi

# Default port
PORT=${1:-6006}

echo "=========================================="
echo "Starting TensorBoard Server"
echo "=========================================="
echo "Log directory: $LOG_DIR"
echo "Port: $PORT"
echo ""
echo "To access TensorBoard:"
echo "  1. If on local machine: http://localhost:$PORT"
echo "  2. If on remote server, create SSH tunnel:"
echo "     ssh -L $PORT:localhost:$PORT <username>@<server>"
echo "     Then open: http://localhost:$PORT in your browser"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo "=========================================="
echo ""

# Launch TensorBoard
tensorboard --logdir="$LOG_DIR" --port="$PORT" --bind_all

