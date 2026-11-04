#!/bin/bash
# Keep monitor_status.json updated by reading log files
# This is a temporary workaround until simulation is restarted with updated plugin

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="/work/mh0033/m300883/Project_week/MEssE/build/messe_env/py_env"

source "$VENV_PATH/bin/activate"

echo "Starting status updater (updates every 5 seconds)..."
echo "Press Ctrl+C to stop"

while true; do
    python "$SCRIPT_DIR/generate_status.py" > /dev/null 2>&1
    sleep 5
done
