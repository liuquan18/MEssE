#!/usr/bin/env python
"""
Script to generate monitor_status.json from ICON simulation log files.
Supports both GNN and MLP training modes with enhanced metrics.
"""

import json
import glob
import os
from datetime import datetime
import getpass

# Get user-specific scratch directory
user = getpass.getuser()
log_dir = f"/scratch/{user[0]}/{user}/icon_exercise_comin"

# Find all log files and sort by modification time (most recent first)
log_files = glob.glob(os.path.join(log_dir, "log_*.txt"))
log_files.sort(key=os.path.getmtime, reverse=True)

if not log_files:
    print(f"No log files found in {log_dir}")
    exit(1)

# Read the latest log file (most recently modified)
latest_log = log_files[0]
filename = os.path.basename(latest_log)

# Extract timestamp from filename
# Format: log_2021-07-14_00-00-00.txt (updated format with underscores and dashes)
try:
    time_str = filename.replace("log_", "").replace(".txt", "")
    # Try new format first (with underscores and dashes)
    try:
        current_time = datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S")
    except:
        # Fall back to old format
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

# Estimate start time from oldest log file
if len(log_files) > 1:
    oldest_log = log_files[-1]  # Last in reverse sorted list is oldest
    first_filename = os.path.basename(oldest_log)
    try:
        time_str = first_filename.replace("log_", "").replace(".txt", "")
        try:
            start_time = datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S")
        except:
            start_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    except:
        start_time = datetime(2021, 7, 14, 0, 0, 0)
else:
    start_time = datetime(2021, 7, 14, 0, 0, 0)

elapsed_time = current_time - start_time

# Detect model type from checkpoint files
model_type = "Unknown"
num_nodes = 0
try:
    import torch

    checkpoint_files = glob.glob(os.path.join(log_dir, "net_*.pth"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        checkpoint = torch.load(latest_checkpoint, map_location="cpu")
        use_gnn = checkpoint.get("use_gnn", False)
        model_type = "GNN (Mini-batch)" if use_gnn else "MLP"
except:
    # If we can't load checkpoint, estimate from number of losses per file
    if losses:
        if len(losses) > 10:  # GNN typically has more batches per timestep
            model_type = "GNN (Mini-batch)"
        else:
            model_type = "MLP"

# Estimate learning rate based on model type
learning_rate = 0.001 if "GNN" in model_type else 0.01

# Create status data with enhanced information
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
        "model_type": model_type,
        "current_loss": float(losses[-1]) if losses else 0.0,
        "total_batches": len(all_losses),
        "batches_per_timestep": len(losses) if losses else 0,
        "learning_rate": learning_rate,
        "avg_loss": float(sum(all_losses) / len(all_losses)) if all_losses else 0.0,
        "min_loss": float(min(all_losses)) if all_losses else 0.0,
        "max_loss": float(max(all_losses)) if all_losses else 0.0,
    },
}

# Write to JSON file
output_file = os.path.join(log_dir, "monitor_status.json")
with open(output_file, "w") as f:
    json.dump(status_data, f, indent=2)

print(f"Generated {output_file}")
print(f"Model type: {status_data['training']['model_type']}")
print(f"Current loss: {status_data['training']['current_loss']:.6e}")
print(f"Total batches: {status_data['training']['total_batches']}")
print(f"Batches/timestep: {status_data['training']['batches_per_timestep']}")
print(f"Learning rate: {status_data['training']['learning_rate']}")
print(f"Output count: {status_data['simulation']['output_count']}")
