#!/bin/bash
# Usage: bash scripts/monitor.sh JOB_ID PORT
# Then open http://localhost:PORT in your browser (after SSH port-forward if needed)

JOB_ID=${1:?"Usage: $0 JOB_ID PORT"}
PORT=${2:?"Usage: $0 JOB_ID PORT"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/../monitor/app.py" "$JOB_ID" "$PORT"
