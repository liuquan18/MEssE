#!/bin/bash
# MEssE v1.0 Monitoring Server Launcher

echo "=============================================="
echo "  MEssE v1.0 - Monitoring Server"
echo "=============================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate Python virtual environment
VENV_PATH="/work/mh0033/m300883/Project_week/MEssE/build/messe_env/py_env"
echo "🐍 Activating Python environment..."
source "$VENV_PATH/bin/activate"

# Check if Flask is installed
if ! python -c "import flask" &> /dev/null; then
    echo "⚠️  Flask is not installed. Installing now..."
    pip install flask
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Flask. Please install manually:"
        echo "   pip install flask"
        exit 1
    fi
    echo "✅ Flask installed successfully"
    echo ""
fi

# Get hostname and port
HOSTNAME=$(hostname)
PORT=${MONITOR_PORT:-5000}

echo "🚀 Starting monitoring server..."
echo ""
echo "📡 Access URLs:"
echo "   Local:    http://localhost:$PORT"
echo "   Network:  http://$HOSTNAME:$PORT"
echo ""
echo "💡 Press Ctrl+C to stop the server"
echo "=============================================="
echo ""

# Start the server
cd "$SCRIPT_DIR"
python monitor_server.py $PORT
