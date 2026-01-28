#!/usr/bin/env python
"""
Temporary script to generate monitor_status.json from existing log files
until the simulation is restarted with the updated comin_plugin.py
"""

import json
import glob
import os
from datetime import datetime

# Directory with log files
log_dir = "/scratch/m/m300883/icon_exercise_comin"

# Find all log files and sort by modification time (most recent first)
log_files = glob.glob(os.path.join(log_dir, "log_*.txt"))
log_files.sort(key=os.path.getmtime, reverse=True)

if not log_files:
    print("No log files found")
    exit(1)

# Read the latest log file (most recently modified)
latest_log = log_files[0]
filename = os.path.basename(latest_log)

# Extract timestamp from filename
# Format: log_2021-07-14 00:00:00.txt
try:
    time_str = filename.replace("log_", "").replace(".txt", "")
    current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
except:
    current_time = datetime.now()

# Read losses from the log file
losses = []
try:
    with open(latest_log, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    losses.append(float(line))
                except:
                    pass
except:
    pass

# Get all losses from recent files (last 10 most recently modified)
all_losses = []
recent_files = sorted(log_files[:10], key=os.path.getmtime, reverse=True)
for log_file in recent_files:
    try:
        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_losses.append(float(line))
                    except:
                        pass
    except:
        pass

# Take last 100 losses
all_losses = all_losses[-100:]

# Estimate start time (assuming hourly outputs based on filenames)
if len(log_files) > 1:
    first_log = log_files[0]
    first_filename = os.path.basename(first_log)
    try:
        time_str = first_filename.replace("log_", "").replace(".txt", "")
        start_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    except:
        start_time = datetime(2021, 7, 14, 0, 0, 0)
else:
    start_time = datetime(2021, 7, 14, 0, 0, 0)

elapsed_time = current_time - start_time

# Create status data
status_data = {
    "timestamp": current_time.isoformat(),
    "simulation": {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_time": str(elapsed_time),
        "n_domains": 1,
        "total_points": len(losses) * 5 * 30 if losses else 0,  # Estimated
        "output_count": len(log_files),
    },
    "training": {
        "current_loss": float(losses[-1]) if losses else 0.0,
        "total_batches": len(all_losses),
        "learning_rate": 0.01,
        "avg_loss": float(sum(all_losses) / len(all_losses)) if all_losses else 0.0,
        "min_loss": float(min(all_losses)) if all_losses else 0.0,
    },
}

# Write to JSON file
output_file = os.path.join(log_dir, "monitor_status.json")
with open(output_file, "w") as f:
    json.dump(status_data, f, indent=2)

print(f"Generated {output_file}")
print(f"Current loss: {status_data['training']['current_loss']:.6f}")
print(f"Total batches: {status_data['training']['total_batches']}")
print(f"Output count: {status_data['simulation']['output_count']}")
