#!/usr/bin/env python
"""
MEssE v1.0 Monitoring Server
Real-time monitoring of ICON simulation and NN training
"""

import sys
from flask import Flask, render_template, jsonify
import os
import glob
import json
import getpass
from datetime import datetime
import numpy as np

app = Flask(__name__)

user = getpass.getuser()
SCRATCH_DIR = f"/scratch/{user[0]}/{user}/icon_exercise_comin"
# Get the project root directory (parent of monitor/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
EXPERIMENT_DIR = os.path.join(
    PROJECT_ROOT, "build/messe_env/build_dir/icon-model/experiments/esm_bb_ruby0"
)

PORT=sys.argv[1] if len(sys.argv) > 1 else "5005"

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
    """Get loss history from log files with timestamps"""
    log_files = glob.glob(os.path.join(SCRATCH_DIR, "log_*.txt"))
    if not log_files:
        return []

    # Sort by filename (which contains timestamp)
    log_files.sort()

    loss_data = []
    for log_file in log_files[-100:]:  # Last 100 timesteps
        try:
            # Extract timestamp from filename
            filename = os.path.basename(log_file)
            time_str = filename.replace("log_", "").replace(".txt", "")

            # Parse timestamp - format is YYYY-MM-DDTHH-MM-SS.sss
            # Convert to more standard format for parsing
            time_str_clean = time_str.replace("T", " ").replace(
                "-", ":", 2
            )  # Replace first 2 dashes with colons
            time_str_clean = time_str_clean.rsplit("-", 1)[
                0
            ]  # Remove milliseconds part

            # Read loss from file (should be single value - average of batches)
            with open(log_file, "r") as f:
                loss_line = f.readline().strip()
                if loss_line:
                    loss = float(loss_line)
                    loss_data.append(
                        {
                            "time": time_str,  # Keep original format for display
                            "loss": loss,
                        }
                    )
        except Exception as e:
            pass

    return loss_data


def get_temperature_history():
    """Get global mean temperature from NetCDF files"""
    try:
        import netCDF4 as nc
    except ImportError:
        return []

    nc_files = glob.glob(os.path.join(EXPERIMENT_DIR, "esm_bb_ruby0_atm_mon_*.nc"))
    if not nc_files:
        return []

    # Sort by filename
    nc_files.sort()

    temp_data = []
    for nc_file in nc_files[-100:]:  # Last 100 timesteps
        try:
            # Extract timestamp from filename
            filename = os.path.basename(nc_file)
            # Format: esm_bb_ruby0_atm_mon_19790101T180000Z.nc
            time_str = (
                filename.replace("esm_bb_ruby0_atm_mon_", "")
                .replace(".nc", "")
                .replace("Z", "")
            )

            # Open NetCDF file and read tas_gmean
            with nc.Dataset(nc_file, "r") as dataset:
                if "tas_gmean" in dataset.variables:
                    tas_gmean = dataset.variables["tas_gmean"][:]
                    # Take mean over spatial dimensions (should already be global mean)
                    # tas_gmean is (time, lat, lon) but for gmean might be scalar or need averaging
                    if tas_gmean.size > 0:
                        temp_value = float(np.mean(tas_gmean))  # Take mean if needed
                        temp_data.append({"time": time_str, "temperature": temp_value})
        except Exception as e:
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
    temperatures = get_temperature_history()

    if status is None:
        return jsonify(
            {
                "status": "waiting",
                "message": "Waiting for simulation data...",
                "losses": [],
                "temperatures": [],
            }
        )

    return jsonify(
        {
            "status": "running",
            "simulation": status.get("simulation", {}),
            "training": status.get("training", {}),
            "losses": losses,
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

    app.run(host="0.0.0.0", port=int(PORT), debug=False)
