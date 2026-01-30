#!/usr/bin/env python3
"""
Simple script to visualize diffusion model training loss evolution.
Reads log files from /scratch/m/m301257/icon_exercise_comin/
"""

import glob
import re
import matplotlib.pyplot as plt
from datetime import datetime
import getpass

# Get log files (both old format log_*.txt and new format log_session_*.txt)
user = getpass.getuser()
log_dir = f"/scratch/{user[0]}/{user}/icon_exercise_comin"
log_files = sorted(glob.glob(f"{log_dir}/log_session_*.txt"))

# If new format not found, try old format
if not log_files:
    log_files = sorted(glob.glob(f"{log_dir}/log_*.txt"))
    print("Using old log format")
else:
    print("Using new log format (log_session_*)")

if not log_files:
    print(f"No log files found in {log_dir}")
    exit(1)

print(f"Found {len(log_files)} training sessions")

# Parse logs
training_sessions = []
mean_losses = []
timestamps = []

for log_file in log_files:
    filename = log_file.split('/')[-1]
    
    # Read file content
    session_num = None
    time_str = None
    epoch_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # New format: training_session: 1
            if line.startswith('training_session:'):
                session_num = int(line.split(':')[1].strip())
            # New format: simulation_time: 1979-01-03T12-00-00
            elif line.startswith('simulation_time:'):
                time_str = line.split(':',1)[1].strip()
            # Both formats: epoch_1: 0.123456
            elif 'epoch_' in line:
                match = re.search(r'epoch_\d+:\s+([\d.]+)', line)
                if match:
                    epoch_losses.append(float(match.group(1)))
    
    # Extract timestamp
    if time_str:
        # New format: simulation_time field
        timestamp = datetime.strptime(time_str, "%Y-%m-%dT%H-%M-%S")
    else:
        # Old format: from filename log_1979-01-03T12-00-00.txt
        time_str = filename.replace('log_', '').replace('.txt', '')
        # Handle log_session_0001_1979-01-03T12-00-00.txt format
        if '_' in time_str:
            time_str = time_str.split('_')[-1]  # Get last part after underscore
        timestamp = datetime.strptime(time_str, "%Y-%m-%dT%H-%M-%S")
    
    if epoch_losses:
        training_sessions.append(session_num if session_num else len(training_sessions) + 1)
        timestamps.append(timestamp)
        mean_losses.append(sum(epoch_losses) / len(epoch_losses))

if not timestamps:
    print("No loss data found in log files")
    exit(1)

# Calculate relative timesteps (in hours from start)
start_time = timestamps[0]
hours = [(t - start_time).total_seconds() / 3600 for t in timestamps]

# Create plot with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Loss vs Time (hours)
ax1.plot(hours, mean_losses, 'o-', linewidth=2, markersize=8, color='#2E86AB')
ax1.set_xlabel('Training Time (hours)', fontsize=14)
ax1.set_ylabel('Mean Loss (MSE)', fontsize=14)
ax1.set_title('Loss Evolution Over Time', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Loss vs Training Session Number
ax2.plot(training_sessions, mean_losses, 'o-', linewidth=2, markersize=8, color='#A23B72')
ax2.set_xlabel('Training Session', fontsize=14)
ax2.set_ylabel('Mean Loss (MSE)', fontsize=14)
ax2.set_title('Loss Evolution by Session', fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.suptitle('Diffusion Model Training (Continuous after Buffer Full)', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()

# Save plot
output_file = f"/work/mh1498/m301257/project2/MEssE/scripts/validate/loss_evolution.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Plot saved to: {output_file}")

# Print summary
print(f"\nTraining Summary:")
print(f"  Sessions: {len(timesteps)}")
print(f"  Duration: {hours[-1]:.1f} hours")
print(f"  Initial loss: {mean_losses[0]:.6f}")
print(f"  Final loss: {mean_losses[-1]:.6f}")
print(f"  Best loss: {min(mean_losses):.6f}")

# Show plot
plt.show()
