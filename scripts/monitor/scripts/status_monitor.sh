#!/bin/bash
# Check the status of the ICON ML Training Loss Monitor Dashboard

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MONITOR_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$MONITOR_DIR/dashboard.pid"

echo -e "${BLUE}ICON ML Training Loss Monitor - Status${NC}"
echo "========================================"
echo ""

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo -e "Status: ${RED}NOT RUNNING${NC}"
    echo "PID file not found: $PID_FILE"
    exit 0
fi

PID=$(cat "$PID_FILE")

# Check if process is running
if ps -p $PID > /dev/null 2>&1; then
    echo -e "Status: ${GREEN}RUNNING${NC}"
    echo "PID: $PID"
    echo ""
    
    # Get process info
    echo "Process Information:"
    ps -p $PID -o pid,ppid,%cpu,%mem,etime,command
    echo ""
    
    # Check log file
    if [ -f "$MONITOR_DIR/dashboard.log" ]; then
        echo "Recent log entries:"
        echo "-------------------"
        tail -n 10 "$MONITOR_DIR/dashboard.log"
    fi
    
    echo ""
    echo -e "${GREEN}Dashboard is accessible at:${NC}"
    # Try to extract port from log
    if [ -f "$MONITOR_DIR/dashboard.log" ]; then
        PORT=$(grep -oP "Running on.*:(\d+)" "$MONITOR_DIR/dashboard.log" | tail -1 | grep -oP "\d+$" || echo "5000")
        echo -e "  ${BLUE}http://localhost:$PORT${NC}"
    fi
else
    echo -e "Status: ${RED}NOT RUNNING${NC}"
    echo "PID file exists but process $PID is not running"
    echo "Cleaning up stale PID file..."
    rm -f "$PID_FILE"
fi

echo ""

# Check data directory
USER=$(whoami)
DATA_DIR="/scratch/${USER:0:1}/$USER/icon_exercise_comin"

echo "Data Directory: $DATA_DIR"
if [ -d "$DATA_DIR" ]; then
    LOG_COUNT=$(find "$DATA_DIR" -name "log_*.txt" 2>/dev/null | wc -l)
    SUMMARY_COUNT=$(find "$DATA_DIR" -name "summary_*.txt" 2>/dev/null | wc -l)
    echo -e "  ${GREEN}✓${NC} Directory exists"
    echo "  Log files: $LOG_COUNT"
    echo "  Summary files: $SUMMARY_COUNT"
    
    # Show most recent files
    if [ $LOG_COUNT -gt 0 ]; then
        echo ""
        echo "Most recent log files:"
        ls -lt "$DATA_DIR"/log_*.txt 2>/dev/null | head -3 | awk '{print "  " $9}'
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Directory does not exist yet"
fi
