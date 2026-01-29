"""
Inference script for GNN model on ICON grid data.
This script:
1. Loads data from native ICON grid NetCDF files
2. Loads trained GNN model
3. Performs inference in mini-batches
4. Compares predictions with ICON simulation results
"""

# %%
import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import getpass

# Import model and utilities
from gnn_model import SimpleGNN
from utils import process_batches_for_inference, load_model_checkpoint

# %%
# Configuration
user = getpass.getuser()
SCRATCH_DIR = Path(f"/scratch/{user[0]}/{user}/icon_exercise_comin")
EXPERIMENT_DIR = Path("/work/mh0033/m300883/Project_week/MEssE/experiment")

# Model configuration (should match training)
MODEL_CONFIG = {
    "in_channels": 1,
    "hidden_channels": 32,
    "out_channels": 1,
    "num_layers": 3,
}

BATCH_SIZE = 5000
K_NEIGHBORS = 6
EXTENDED_K = 8


def load_icon_data(filepath):
    """
    Load data from native ICON grid NetCDF file.

    Parameters:
    -----------
    filepath : str or Path
        Path to NetCDF file

    Returns:
    --------
    data : xr.Dataset
        Loaded dataset
    """
    print(f"Loading data from {filepath}")
    data = xr.open_dataset(filepath)
    print(f"  Variables: {list(data.data_vars)}")
    print(f"  Dimensions: {dict(data.dims)}")
    return data


def extract_coordinates(data):
    """
    Extract node coordinates from ICON grid data.

    Parameters:
    -----------
    data : xr.Dataset
        ICON dataset

    Returns:
    --------
    pos : np.ndarray
        Node positions [num_nodes, 2] in degrees (lon, lat)
    """
    # ICON grid stores coordinates in radians
    if "clon" in data.coords and "clat" in data.coords:
        lon = np.rad2deg(data["clon"].values)
        lat = np.rad2deg(data["clat"].values)
    elif "lon" in data.coords and "lat" in data.coords:
        lon = data["lon"].values
        lat = data["lat"].values
    else:
        raise ValueError("Could not find longitude/latitude coordinates in dataset")

    pos = np.column_stack([lon, lat])
    print(f"Extracted coordinates: {pos.shape}")
    return pos


# %%
# Find available data files
data_files = sorted(glob.glob(str(EXPERIMENT_DIR / "NWP_LAM_DOM01_*.nc")))
print(f"Found {len(data_files)} data files")

# Find the latest checkpoint
checkpoint_files = sorted(glob.glob(str(SCRATCH_DIR / "net_*.pth")))


checkpoint_file = checkpoint_files[-1]  # Use the latest checkpoint
print(f"Using checkpoint: {checkpoint_file}")
# %%
data_file = data_files[1]
# read data
data = load_icon_data(data_file)

# %%
# 2. Extract coordinates
pos = extract_coordinates(data)
num_nodes = pos.shape[0]
print(f"Total nodes: {num_nodes}")

# 3. Extract input variable (RHI_MAX)
if "RHI_MAX" not in data.data_vars:
    raise ValueError("RHI_MAX not found in dataset")

rhi_max = data["RHI_MAX"].isel(time=1).values
if len(rhi_max.shape) > 1:
    rhi_max = rhi_max.squeeze()

print(
    f"Input (RHI_MAX): shape={rhi_max.shape}, min={rhi_max.min():.2f}, max={rhi_max.max():.2f}"
)

# 4. Extract ground truth (QI_MAX) if available
has_ground_truth = "QI_MAX" in data.data_vars
if has_ground_truth:
    qi_max = data["QI_MAX"].values
    if len(qi_max.shape) > 1:
        qi_max = qi_max.squeeze()
    print(
        f"Ground truth (QI_MAX): shape={qi_max.shape}, min={qi_max.min():.2e}, max={qi_max.max():.2e}"
    )
else:
    print("Warning: QI_MAX not found in dataset, skipping comparison")
    qi_max = None

# 5. Load model
print(f"\nLoading model from {checkpoint_file}")
model, checkpoint = load_model_checkpoint(checkpoint_file, SimpleGNN, MODEL_CONFIG)
print(f"  Model type: {checkpoint.get('use_gnn', 'GNN')}")
num_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {num_params:,}")

# 6. Perform inference in batches
print(f"\nPerforming inference...")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Number of batches: {(num_nodes + BATCH_SIZE - 1) // BATCH_SIZE}")

predictions = process_batches_for_inference(
    data=rhi_max,
    pos=pos,
    model=model,
    batch_size=BATCH_SIZE,
    k=K_NEIGHBORS,
    extended_k=EXTENDED_K,
    device="cpu",
)

predictions = predictions.squeeze()
print(
    f"Predictions: shape={predictions.shape}, min={predictions.min():.2e}, max={predictions.max():.2e}"
)
# %%
# back to xarray data array
rhi_max_xr = data.isel(time=1)["RHI_MAX"].squeeze()
qi_max_xr = data.isel(time=1)["QI_MAX"].squeeze() if has_ground_truth else None
predictions = qi_max_xr.copy(data=predictions)

# %%
# plot results
fig, axes = plt.subplots(1, 3 if has_ground_truth else 2, figsize=(15, 5))

# Plot 1: Input (RHI_MAX)
rhi_plot = rhi_max_xr.values.copy()
rhi_plot[~np.isfinite(rhi_plot)] = 0  # Replace non-finite with 0
im0 = axes[0].tricontourf(
    rhi_max_xr["clon"].values,
    rhi_max_xr["clat"].values,
    rhi_plot,
    levels=20,
    cmap="Blues",
)
axes[0].set_title("Input: RHI_MAX")
plt.colorbar(im0, ax=axes[0], orientation="vertical", label="RHI_MAX (%)")

# Plot 2: Ground Truth (QI_MAX) if available
if has_ground_truth:
    qi_plot = qi_max_xr.values.copy()
    qi_plot[~np.isfinite(qi_plot)] = 0  # Replace non-finite with 0
    im2 = axes[1].tricontourf(
        qi_max_xr["clon"].values,
        qi_max_xr["clat"].values,
        qi_plot,
        levels=20,
        cmap="Reds",
    )
    axes[1].set_title("Ground Truth: QI_MAX")
    plt.colorbar(im2, ax=axes[1], orientation="vertical", label="QI_MAX (kg/kg)")

# Plot 3: Predictions
pred_plot = predictions.values.copy()
pred_plot[~np.isfinite(pred_plot)] = 0  # Replace non-finite with 0
im1 = axes[2 if has_ground_truth else 1].tricontourf(
    predictions["clon"].values,
    predictions["clat"].values,
    pred_plot,
    levels=20,
    cmap="Reds",
)
axes[2 if has_ground_truth else 1].set_title("Predicted: QI_MAX")
plt.colorbar(
    im1,
    ax=axes[2 if has_ground_truth else 1],
    orientation="vertical",
    label="QI_MAX (kg/kg)",
)

plt.tight_layout()
plt.show()
# %%
