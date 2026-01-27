#!/usr/bin/env python
"""
Plot loss function values saved during ICON simulation with ComIn plugin.
"""
# %%
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import getpass

# %%
# Get username
user = getpass.getuser()

# Directory where log files are saved
log_dir = Path(f"/scratch/{user[0]}/{user}/icon_exercise_comin")

# Find all log files
log_files = sorted(log_dir.glob("log_*.txt"))

if not log_files:
    print(f"No log files found in {log_dir}")
    exit(1)

print(f"Found {len(log_files)} log files")

# Read loss values from each file
data = []

for log_file in log_files:
    # Extract timestamp from filename
    # Format: log_2021-07-14 00:00:00.txt
    filename = log_file.stem  # Remove .txt extension
    timestamp_str = filename.replace("log_", "")
    timestamp = pd.to_datetime(timestamp_str)

    # Read loss values from file
    with open(log_file, "r") as f:
        losses = [float(line.strip()) for line in f if line.strip()]

    # Store timestamp and all losses
    for batch_idx, loss in enumerate(losses):
        data.append({"timestamp": timestamp, "batch": batch_idx, "loss": loss})

# Create DataFrame
df = pd.DataFrame(data)

# Limit to first 200 data points
df = df.head(200)

print(f"Total data points: {len(df)}")
print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Loss vs time (averaged over batches per timestep)
ax1 = axes[0]
avg_loss = df.groupby("timestamp")["loss"].mean()
ax1.plot(avg_loss.index.values, avg_loss.values, marker="o", linestyle="-", linewidth=2)
ax1.set_xlabel("Time", fontsize=12)
ax1.set_ylabel("Average Loss", fontsize=12)
ax1.set_title("Average Loss per Timestep", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis="x", rotation=45)

# Plot 2: All individual batch losses
ax2 = axes[1]
for timestamp in df["timestamp"].unique():
    subset = df[df["timestamp"] == timestamp]
    ax2.plot(
        subset["batch"].values,
        subset["loss"].values,
        marker="o",
        alpha=0.6,
        label=timestamp,
    )

ax2.set_xlabel("Batch Number", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.set_title("Loss per Batch (All Timesteps)", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3)
# Only show legend if not too many timesteps
if len(df["timestamp"].unique()) <= 10:
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

plt.tight_layout()

# Save figure
output_file = log_dir / "loss_function_plot.png"
# plt.savefig(output_file, dpi=300, bbox_inches="tight")
# print(f"Plot saved to: {output_file}")

# Also create a simple time series plot
fig2, ax = plt.subplots(figsize=(12, 6))
ax.scatter(df["timestamp"].values, df["loss"].values, alpha=0.5, s=20)
ax.plot(
    avg_loss.index.values, avg_loss.values, color="red", linewidth=3, label="Average"
)
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Loss Function Over Time (All Batches)", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()

output_file2 = log_dir / "loss_timeseries_plot.png"
plt.savefig(output_file2, dpi=300, bbox_inches="tight")
print(f"Time series plot saved to: {output_file2}")

# Print summary statistics
print("\nSummary Statistics:")
print(
    f"Initial loss (first timestep avg): {df[df['timestamp'] == df['timestamp'].min()]['loss'].mean():.6f}"
)
print(
    f"Final loss (last timestep avg): {df[df['timestamp'] == df['timestamp'].max()]['loss'].mean():.6f}"
)
print(f"Overall average loss: {df['loss'].mean():.6f}")
print(f"Overall std loss: {df['loss'].std():.6f}")
print(f"Min loss: {df['loss'].min():.6f}")
print(f"Max loss: {df['loss'].max():.6f}")

plt.show()

# %%
