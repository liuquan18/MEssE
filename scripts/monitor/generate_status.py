#!/usr/bin/env python
"""
Script to generate monitor_status.json for diffusion model training.
Parses ICON simulation logs to extract training history.
"""

import json
import re
from datetime import datetime
import os
import getpass

# Configuration
user = getpass.getuser()
# Look for SLURM job log files in the working directory
work_dir = "/work/mh1498/m301257/project2/MEssE"
log_pattern = os.path.join(work_dir, "exp.esm_bb_ruby0_*.out")

# Try to find the most recent SLURM output file
import glob
log_files = glob.glob(log_pattern)

if not log_files:
    # Try alternate location
    log_pattern2 = f"/scratch/{user[0]}/{user}/icon_exercise_comin/exp.esm_bb_ruby0_*.out"
    log_files = glob.glob(log_pattern2)

if not log_files:
    print("No log files found. Looking for SLURM output files matching exp.esm_bb_ruby0_*.out")
    status = {
        "status": "waiting",
        "message": "No training data found yet",
        "history": []
    }
    # Write empty status
    output_file = "/work/mh1498/m301257/project2/MEssE/scripts/monitor/status.json"
    with open(output_file, "w") as f:
        json.dump(status, f, indent=2)
    exit(0)

# Use most recent log file
log_file = max(log_files, key=os.path.getmtime)
print(f"Parsing log file: {log_file}")

# Parse training sessions from log
training_history = []
start_time = None

with open(log_file, "r") as f:
    for line in f:
        # Look for training completion lines:
        # "Training complete at 1979-01-03 12:00:00: final loss = 0.684880"
        match = re.search(r'Training complete at ([\d-]+ [\d:]+): final loss = ([\d.]+)', line)
        if match:
            timestamp_str = match.group(1)
            final_loss = float(match.group(2))
            
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                
                # Calculate timestep (simulation starts at some point, we use relative time)
                if start_time is None:
                    start_time = timestamp
                
                timestep = int((timestamp - start_time).total_seconds() / 900)  # 15-minute intervals
                
                training_history.append({
                    'timestep': timestep,
                    'mean_loss': final_loss,
                    'timestamp': timestamp.isoformat()
                })
            except Exception as e:
                print(f"Error parsing timestamp: {e}")
                continue

if not training_history:
    print("No training data found in logs")
    status = {
        "status": "waiting",
        "message": "Simulation running but no training completed yet",
        "history": []
    }
else:
    latest = training_history[-1]
    status = {
        "status": "running",
        "simulation": {
            "current_time": latest['timestamp'],
            "timestep": latest['timestep'],
            "total_training_sessions": len(training_history)
        },
        "training": {
            "latest_loss": latest['mean_loss'],
            "num_epochs": 3,
            "training_interval": "12 timesteps (3 hours)",
            "buffer_size": "48 timesteps (12 hours)"
        },
        "history": training_history
    }

# Write status.json
output_file = "/work/mh1498/m301257/project2/MEssE/scripts/monitor/status.json"
with open(output_file, "w") as f:
    json.dump(status, f, indent=2)

print(f"Generated {output_file}")
print(f"Training sessions found: {len(training_history)}")
if training_history:
    print(f"Latest loss: {latest['mean_loss']:.6f} at timestep {latest['timestep']}")
