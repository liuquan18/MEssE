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

PORT = sys.argv[1] if len(sys.argv) > 1 else 5000

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
    for log_file in log_files[-10:]:  # Last 10 timesteps
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

    return all_epoch_losses[-100:]  # Return last 100 epoch losses


def get_temperature_history():
    """Get global mean temperature from NetCDF files"""
    try:
        import netCDF4 as nc
        import warnings
    except ImportError:
        return []

    # Suppress HDF5 error messages (files may be locked by ICON simulation)
    import os as os_hdf5

    os_hdf5.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    # Suppress warnings from netCDF4/HDF5
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*HDF5.*")

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

            # Open NetCDF file and read tas_gmean (skip if locked)
            with nc.Dataset(nc_file, "r", parallel=False) as dataset:
                if "tas_gmean" in dataset.variables:
                    tas_gmean = dataset.variables["tas_gmean"][:]
                    # Take mean over spatial dimensions (should already be global mean)
                    if tas_gmean.size > 0:
                        temp_value = float(np.mean(tas_gmean))  # Take mean if needed
                        temp_data.append({"time": time_str, "temperature": temp_value})
        except (OSError, IOError, RuntimeError):
            # Skip files that are locked or being written to
            continue
        except Exception:
            # Skip any other errors silently
            pass

    # Skip the first timestep
    if len(temp_data) > 1:
        temp_data = temp_data[1:]

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
    print(f"\n  🌐 Access the interface at: http://localhost:5005")
    print(f"  📁 Data directory: {SCRATCH_DIR}")
    print("\n" + "=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5005, debug=False)
