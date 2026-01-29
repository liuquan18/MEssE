"""
Inference script for GNN model on ICON grid data.
This script:
1. Loads data from native ICON grid NetCDF files
2. Loads trained GNN model
3. Performs inference in mini-batches
4. Compares predictions with ICON simulation results
"""

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


def perform_inference(data_file, checkpoint_file, output_file=None):
    """
    Main inference pipeline.

    Parameters:
    -----------
    data_file : str or Path
        Path to input NetCDF file with native ICON grid data
    checkpoint_file : str or Path
        Path to trained model checkpoint
    output_file : str or Path, optional
        Path to save results

    Returns:
    --------
    results : dict
        Dictionary containing predictions, ground truth, and metrics
    """
    print("=" * 70)
    print("GNN Inference on ICON Grid Data")
    print("=" * 70)

    # 1. Load data
    data = load_icon_data(data_file)

    # 2. Extract coordinates
    pos = extract_coordinates(data)
    num_nodes = pos.shape[0]
    print(f"Total nodes: {num_nodes}")

    # 3. Extract input variable (RHI_MAX)
    if "RHI_MAX" not in data.data_vars:
        raise ValueError("RHI_MAX not found in dataset")

    rhi_max = data["RHI_MAX"].values
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

    # 7. Compare with ground truth
    results = {
        "predictions": predictions,
        "input": rhi_max,
        "coordinates": pos,
    }

    if has_ground_truth:
        results["ground_truth"] = qi_max

        # Calculate metrics
        mse = np.mean((predictions - qi_max) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - qi_max))

        # Calculate relative error (avoid division by zero)
        mask = qi_max != 0
        rel_error = np.zeros_like(qi_max)
        rel_error[mask] = np.abs(predictions[mask] - qi_max[mask]) / np.abs(
            qi_max[mask]
        )
        mean_rel_error = np.mean(rel_error[mask]) if mask.any() else np.nan

        results["metrics"] = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mean_relative_error": float(mean_rel_error),
        }

        print("\n" + "=" * 70)
        print("Evaluation Metrics")
        print("=" * 70)
        print(f"  MSE:  {mse:.6e}")
        print(f"  RMSE: {rmse:.6e}")
        print(f"  MAE:  {mae:.6e}")
        print(f"  Mean Relative Error: {mean_rel_error:.2%}")

    # 8. Save results
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create output dataset
        out_ds = xr.Dataset(
            {
                "RHI_MAX": (["ncells"], rhi_max),
                "QI_MAX_pred": (["ncells"], predictions),
            },
            coords={
                "clon": (
                    ["ncells"],
                    (
                        data["clon"].values
                        if "clon" in data.coords
                        else np.deg2rad(pos[:, 0])
                    ),
                ),
                "clat": (
                    ["ncells"],
                    (
                        data["clat"].values
                        if "clat" in data.coords
                        else np.deg2rad(pos[:, 1])
                    ),
                ),
            },
        )

        if has_ground_truth:
            out_ds["QI_MAX_true"] = (["ncells"], qi_max)
            out_ds["QI_MAX_error"] = (["ncells"], predictions - qi_max)

        # Add attributes
        out_ds["QI_MAX_pred"].attrs = {
            "long_name": "GNN predicted maximum cloud ice content",
            "units": "kg/kg",
        }

        out_ds.to_netcdf(output_file)
        print(f"\nResults saved to {output_file}")

    return results


def plot_comparison(results, output_dir=None):
    """
    Create comparison plots of predictions vs ground truth.

    Parameters:
    -----------
    results : dict
        Results from perform_inference
    output_dir : str or Path, optional
        Directory to save plots
    """
    if "ground_truth" not in results:
        print("Cannot create comparison plot: no ground truth available")
        return

    predictions = results["predictions"]
    ground_truth = results["ground_truth"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Ground truth
    im0 = axes[0].scatter(
        results["coordinates"][:, 0],
        results["coordinates"][:, 1],
        c=ground_truth,
        s=1,
        cmap="viridis",
    )
    axes[0].set_title("Ground Truth (QI_MAX)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im0, ax=axes[0])

    # Predictions
    im1 = axes[1].scatter(
        results["coordinates"][:, 0],
        results["coordinates"][:, 1],
        c=predictions,
        s=1,
        cmap="viridis",
    )
    axes[1].set_title("GNN Predictions")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[1])

    # Error
    error = predictions - ground_truth
    im2 = axes[2].scatter(
        results["coordinates"][:, 0],
        results["coordinates"][:, 1],
        c=error,
        s=1,
        cmap="RdBu_r",
        vmin=-np.abs(error).max(),
        vmax=np.abs(error).max(),
    )
    axes[2].set_title("Error (Pred - Truth)")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_file = output_dir / "comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to {plot_file}")

    plt.show()


def main():
    """
    Main function to run inference on multiple files or single file.
    """
    # Find available data files
    data_files = sorted(glob.glob(str(EXPERIMENT_DIR / "NWP_LAM_DOM01_*.nc")))

    if not data_files:
        print(f"No data files found in {EXPERIMENT_DIR}")
        return

    print(f"Found {len(data_files)} data files")

    # Find the latest checkpoint
    checkpoint_files = sorted(glob.glob(str(SCRATCH_DIR / "net_*.pth")))

    if not checkpoint_files:
        print(f"No checkpoint files found in {SCRATCH_DIR}")
        return

    checkpoint_file = checkpoint_files[-1]  # Use the latest checkpoint
    print(f"Using checkpoint: {checkpoint_file}")

    # Process the first file (or you can loop through all files)
    data_file = data_files[0]
    output_file = SCRATCH_DIR / f"inference_result_{Path(data_file).stem}.nc"

    # Run inference
    results = perform_inference(data_file, checkpoint_file, output_file)

    # Create comparison plot if ground truth is available
    if "ground_truth" in results:
        plot_comparison(results, output_dir=SCRATCH_DIR)


if __name__ == "__main__":
    main()
