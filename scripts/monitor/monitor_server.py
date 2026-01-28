#!/usr/bin/env python
"""
MEssE v1.0 Monitoring Server
Real-time monitoring of ICON simulation and NN training
"""

from flask import Flask, render_template, jsonify
import os
import glob
import json
import getpass
from datetime import datetime

app = Flask(__name__)

user = getpass.getuser()
SCRATCH_DIR = f"/scratch/{user[0]}/{user}/icon_exercise_comin"


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
    """Get loss history from log files"""
    log_files = glob.glob(os.path.join(SCRATCH_DIR, "log_*.txt"))
    if not log_files:
        return []

    # Sort by modification time
    log_files.sort(key=os.path.getmtime)

    all_losses = []
    for log_file in log_files[-10:]:  # Last 10 files
        try:
            with open(log_file, "r") as f:
                losses = [float(line.strip()) for line in f if line.strip()]
                all_losses.extend(losses)
        except:
            pass

    return all_losses[-100:]  # Return last 100 loss values


@app.route("/")
def index():
    """Render the main monitoring page"""
    return render_template("monitor.html")


@app.route("/api/status")
def api_status():
    """API endpoint for current status"""
    status = get_latest_status()
    losses = get_loss_history()

    if status is None:
        return jsonify(
            {
                "status": "waiting",
                "message": "Waiting for simulation data...",
                "losses": [],
            }
        )

    return jsonify(
        {
            "status": "running",
            "simulation": status.get("simulation", {}),
            "training": status.get("training", {}),
            "losses": losses,
            "timestamp": status.get("timestamp", ""),
        }
    )


if __name__ == "__main__":
    # Ensure scratch directory exists
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  MEssE v1.0 - Monitoring Server Starting...")
    print("=" * 60)
    print(f"\n  🌐 Access the interface at: http://localhost:5000")
    print(f"  📁 Data directory: {SCRATCH_DIR}")
    print("\n" + "=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False)
