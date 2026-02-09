#!/usr/bin/env python
"""
MEssE v1.0 Monitoring Server
Real-time monitoring of ICON simulation and NN training
"""

from flask import Flask, render_template, jsonify
import os
import sys
import glob
import json
import getpass
import numpy as np
from datetime import datetime

# PORT must be provided as command line argument
if len(sys.argv) < 2:
    print("Error: PORT number is required")
    print("Usage: python monitor_server.py <PORT>")
    print("Example: python monitor_server.py 5005")
    sys.exit(1)
# PORT must be provided as command line argument
if len(sys.argv) < 2:
    print("Error: PORT number is required")
    print("Usage: python monitor_server.py <PORT>")
    print("Example: python monitor_server.py 5005")
    sys.exit(1)

PORT = int(sys.argv[1])

app = Flask(__name__)

user = getpass.getuser()
SCRATCH_DIR = f"/scratch/{user[0]}/{user}/icon_exercise_comin"

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct relative path to experiment directory
EXPERIMENT_DIR = os.path.join(
    SCRIPT_DIR, "../build/messe_env/build_dir/icon-model/experiments/esm_bb_ruby0"
)
EXPERIMENT_DIR = os.path.abspath(EXPERIMENT_DIR)


def get_latest_status():
    """Read the latest status file"""
    status_file = os.path.join(SCRATCH_DIR, "monitor_status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except:
            pass
    return None


def get_loss_history():
    """Get loss history from log files (timestep-level final epoch average)"""
    log_files = glob.glob(os.path.join(SCRATCH_DIR, "log_*.txt"))
    if not log_files:
        return []

    # Sort by modification time
    log_files.sort(key=os.path.getmtime)

    loss_data = []
    for log_file in log_files[-100:]:  # Last 100 timesteps
        try:
            # Extract timestamp from filename
            filename = os.path.basename(log_file)
            time_str = filename.replace("log_", "").replace(".txt", "")

            with open(log_file, "r") as f:
                # Read the final epoch average loss for this timestep
                loss_value = float(f.readline().strip())
                loss_data.append({"time": time_str, "loss": loss_value})
        except:
            pass

    return loss_data


def get_epoch_loss_history():
    """Get detailed epoch-level loss history from detailed log files"""
    log_files = glob.glob(os.path.join(SCRATCH_DIR, "log_detailed_*.txt"))
    if not log_files:
        return []

    # Sort by modification time
    log_files.sort(key=os.path.getmtime)

    all_epoch_losses = []
    for log_file in log_files[-100:]:  # Last 100 timesteps (increased from 10)
        try:
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Parse "Epoch X: loss_value" format
                        if ":" in line:
                            loss_str = line.split(":")[1].strip()
                            all_epoch_losses.append(float(loss_str))
        except:
            pass

    return all_epoch_losses[
        -200:
    ]  # Return last 200 epoch losses (100 timesteps * 2 epochs)


def get_temperature_history():
    """Get global mean temperature from .txt files saved by gnn_trainer"""
    temp_files = glob.glob(os.path.join(SCRATCH_DIR, "global_mean_tas_*.txt"))
    if not temp_files:
        return []

    # Sort by modification time
    temp_files.sort(key=os.path.getmtime)

    temp_data = []
    for temp_file in temp_files[-100:]:  # Last 100 timesteps
        try:
            # Extract timestamp from filename
            filename = os.path.basename(temp_file)
            time_str = filename.replace("global_mean_tas_", "").replace(".txt", "")

            with open(temp_file, "r") as f:
                temp_value = float(f.readline().strip())
                temp_data.append({"time": time_str, "temperature": temp_value})
        except:
            pass

    return temp_data


@app.route("/")
def index():
    """Render the main monitoring page"""
    return render_template("monitor.html")


@app.route("/api/status")
def api_status():
    """API endpoint for current status"""
    status = get_latest_status()
    losses = get_loss_history()
    epoch_losses = get_epoch_loss_history()
    temperatures = get_temperature_history()

    if status is None:
        return jsonify(
            {
                "status": "waiting",
                "message": "Waiting for simulation data...",
                "losses": [],
                "epoch_losses": [],
                "temperatures": [],
            }
        )

    return jsonify(
        {
            "status": "running",
            "simulation": status.get("simulation", {}),
            "hardware": status.get("hardware", {}),
            "training": status.get("training", {}),
            "losses": losses,
            "epoch_losses": epoch_losses,
            "temperatures": temperatures,
            "timestamp": status.get("timestamp", ""),
        }
    )


if __name__ == "__main__":
    # Ensure scratch directory exists
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  MEssE v1.0 - Monitoring Server Starting...")
    print("=" * 60)
    print(f"\n  🌐 Access the interface at: http://localhost:{PORT}")
    print(f"  📁 Data directory: {SCRATCH_DIR}")
    print("\n" + "=" * 60 + "\n")

    app.run(host="0.0.0.0", port=PORT, debug=False)
