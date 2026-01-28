#!/bin/bash
# Start the ICON ML Training Loss Monitor Dashboard

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MONITOR_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     ICON ML Training Loss Monitor - Startup Script          ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration
DEFAULT_PORT=5000
DEFAULT_HOST="0.0.0.0"
PORT="${1:-$DEFAULT_PORT}"
HOST="${2:-$DEFAULT_HOST}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python 3 found: $(python3 --version)"

# Check required Python packages
echo ""
echo "Checking Python dependencies..."

REQUIRED_PACKAGES=("flask" "flask_cors" "pandas" "numpy")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $package installed"
    else
        echo -e "${RED}✗${NC} $package missing"
        MISSING_PACKAGES+=("$package")
    fi
done

# Check optional packages
echo ""
echo "Checking optional dependencies..."
if python3 -c "import tensorboard" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} tensorboard installed (TensorBoard support enabled)"
else
    echo -e "${YELLOW}⚠${NC} tensorboard missing (TensorBoard support disabled)"
fi

# Install missing packages if needed
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Missing required packages: ${MISSING_PACKAGES[*]}${NC}"
    read -p "Install missing packages? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing packages..."
        # Detect if we're in a virtual environment
        if [[ -n "$VIRTUAL_ENV" ]] || [[ -n "$CONDA_DEFAULT_ENV" ]]; then
            echo "Virtual environment detected, installing without --user flag"
            pip install "${MISSING_PACKAGES[@]}"
        else
            echo "Installing to user site-packages"
            pip install --user "${MISSING_PACKAGES[@]}"
        fi
        if [ $? -ne 0 ]; then
            echo -e "${RED}Error: Failed to install packages${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓${NC} Packages installed successfully"
    else
        echo -e "${RED}Cannot start without required packages${NC}"
        exit 1
    fi
fi

# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}Warning: Port $PORT is already in use${NC}"
    read -p "Try a different port? (Enter port number or 'n' to exit): " NEW_PORT
    if [[ $NEW_PORT =~ ^[0-9]+$ ]]; then
        PORT=$NEW_PORT
    else
        echo -e "${RED}Exiting...${NC}"
        exit 1
    fi
fi

# Get username and check data directory
USER=$(whoami)
DATA_DIR="/scratch/${USER:0:1}/$USER/icon_exercise_comin"

echo ""
echo "Monitoring directory: $DATA_DIR"

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${YELLOW}⚠ Warning: Data directory does not exist yet${NC}"
    echo "The dashboard will start but show no data until ICON simulation begins"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Exiting...${NC}"
        exit 1
    fi
else
    LOG_COUNT=$(find "$DATA_DIR" -name "log_*.txt" 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} Data directory exists with $LOG_COUNT log files"
fi

# Create PID file directory
PID_FILE="$MONITOR_DIR/dashboard.pid"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}Dashboard is already running (PID: $OLD_PID)${NC}"
        read -p "Stop it and restart? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kill $OLD_PID 2>/dev/null || true
            sleep 2
        else
            echo -e "${RED}Exiting...${NC}"
            exit 1
        fi
    fi
fi

# Start the dashboard
echo ""
echo -e "${GREEN}Starting dashboard server...${NC}"
echo "Host: $HOST"
echo "Port: $PORT"
echo ""

cd "$MONITOR_DIR"

# Start in background and save PID
nohup python3 dashboard_server.py --host "$HOST" --port "$PORT" > dashboard.log 2>&1 &
DASHBOARD_PID=$!

# Save PID
echo $DASHBOARD_PID > "$PID_FILE"

# Wait a moment and check if it started successfully
sleep 2

if ps -p $DASHBOARD_PID > /dev/null 2>&1; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              Dashboard Started Successfully!                  ║${NC}"
    echo -e "${GREEN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║${NC}  Access the dashboard at:                                    ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}    ${BLUE}http://localhost:$PORT${NC}                                  ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}    ${BLUE}http://$HOST:$PORT${NC}                           ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}                                                               ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}  Process ID: $DASHBOARD_PID                                         ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}  Log file: $MONITOR_DIR/dashboard.log              ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}                                                               ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}  To stop the dashboard, run:                                 ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}    ${YELLOW}$MONITOR_DIR/scripts/stop_monitor.sh${NC}             ${GREEN}║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Monitoring logs in real-time (Ctrl+C to exit log viewer):${NC}"
    echo ""
    tail -f "$MONITOR_DIR/dashboard.log"
else
    echo -e "${RED}Error: Dashboard failed to start${NC}"
    echo "Check the log file: $MONITOR_DIR/dashboard.log"
    rm -f "$PID_FILE"
    exit 1
fi
