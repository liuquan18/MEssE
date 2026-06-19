#!/bin/bash
# Usage: bash scripts/monitor.sh LOG_FILE PORT
# Then open http://localhost:PORT in your browser (after SSH port-forward if needed)

LOG_FILE=${1:?"Usage: $0 LOG_FILE PORT"}
PORT=${2:?"Usage: $0 LOG_FILE PORT"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/../monitor/app.py" "$LOG_FILE" "$PORT"
