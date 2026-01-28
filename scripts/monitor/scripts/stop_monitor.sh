#!/bin/bash
# Stop the ICON ML Training Loss Monitor Dashboard

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MONITOR_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$MONITOR_DIR/dashboard.pid"

echo -e "${YELLOW}Stopping ICON ML Training Loss Monitor...${NC}"

if [ ! -f "$PID_FILE" ]; then
    echo -e "${RED}Dashboard is not running (PID file not found)${NC}"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p $PID > /dev/null 2>&1; then
    echo -e "${RED}Dashboard process (PID: $PID) is not running${NC}"
    rm -f "$PID_FILE"
    exit 1
fi

echo "Sending shutdown signal to process $PID..."
kill $PID

# Wait for process to stop
TIMEOUT=10
COUNTER=0
while ps -p $PID > /dev/null 2>&1 && [ $COUNTER -lt $TIMEOUT ]; do
    sleep 1
    COUNTER=$((COUNTER+1))
    echo -n "."
done
echo ""

if ps -p $PID > /dev/null 2>&1; then
    echo -e "${YELLOW}Process did not stop gracefully, forcing...${NC}"
    kill -9 $PID
    sleep 1
fi

if ps -p $PID > /dev/null 2>&1; then
    echo -e "${RED}Failed to stop dashboard${NC}"
    exit 1
else
    rm -f "$PID_FILE"
    echo -e "${GREEN}✓ Dashboard stopped successfully${NC}"
fi
