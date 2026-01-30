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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_latest_status():
    """Read the latest status file"""
    status_file = os.path.join(SCRIPT_DIR, "status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except:
            pass
    return None


def get_training_history():
    """Get training history with timesteps from status file"""
    status = get_latest_status()
    if status and 'history' in status:
        return status['history']
    return []


@app.route("/")
def index():
    """Render the main monitoring page"""
    return render_template("monitor.html")


@app.route("/api/status")
def api_status():
    """API endpoint for current status"""
    status = get_latest_status()
    history = get_training_history()

    if status is None:
        return jsonify(
            {
                "status": "waiting",
                "message": "Waiting for simulation data...",
                "history": [],
            }
        )

    return jsonify(
        {
            "status": "running",
            "simulation": status.get("simulation", {}),
            "training": status.get("training", {}),
            "history": history,
            "timestamp": status.get("timestamp", ""),
        }
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MEssE v1.0 - Monitoring Server Starting...")
    print("=" * 60)
    print(f"\n  🌐 Access the interface at: http://localhost:5000")
    print(f"  📁 Data directory: {SCRIPT_DIR}")
    print(f"  📊 Status file: {os.path.join(SCRIPT_DIR, 'status.json')}")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False)
