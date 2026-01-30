#!/usr/bin/env python3
"""
Test diffusion model downscaling capability on ICON test data.

This script:
1. Loads trained diffusion model checkpoint
2. Loads test data (coarse R2B01 context + fine R2B04 ground truth)
3. Generates fine-resolution predictions using the model
4. Visualizes coarse input, ground truth, and prediction on global maps
5. Computes evaluation metrics (RMSE, MAE, bias)

Usage:
    python analyze_test_data.py \
        --checkpoint /scratch/m/m301250/icon_exercise_comin/net_1979-01-01T00-00-00.pth \
        --test-dir /scratch/m/m301250/icon_exercise_comin/test_data_1979-01-01T00-00-00 \
        --visualize --n-samples 5
"""

import numpy as np
import glob
import os
import sys
import argparse
from pathlib import Path
import torch

# Import the diffusion model (standalone version without comin dependency)
from diffusion_model import ConditionalDiffusionDownscaler


def load_model(checkpoint_path, device='cpu'):
    """Load trained diffusion model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    
    model = ConditionalDiffusionDownscaler(
        n_fine_cells=256,
        n_neighbors=7,
        n_hidden=256,
        n_timesteps=50
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    print("✅ Model loaded successfully")
    return model, checkpoint


def generate_predictions(model, test_file, device='cpu', n_samples=10):
    """
    Generate predictions for one test sample using the diffusion model.
    
    Returns:
        coarse_data: (80,) coarse center values
        fine_gt: (80, 256) ground truth fine field
        fine_pred: (80, 256) predicted fine field (mean over n_samples)
        timestamp: simulation time string
    """
    # Load test data
    data = np.load(test_file)
    coarse_context_norm = torch.FloatTensor(data['coarse_context_norm']).to(device)  # (80, 7)
    fine_gt = data['fine_field']  # (80, 256)
    coarse_data = data['coarse_context'][:, 0]  # (80,) center values
    timestamp = str(data['simulation_time'])
    
    # Generate predictions
    with torch.no_grad():
        predictions_norm = model.sample(coarse_context_norm, n_samples=n_samples)  # (80, n_samples, 256)
    
    # Denormalize predictions
    predictions_norm = predictions_norm.cpu().numpy()
    ema_fine_mean = float(data['ema_fine_mean'])
    ema_fine_std = float(data['ema_fine_std'])
    predictions = predictions_norm * ema_fine_std + ema_fine_mean  # (80, n_samples, 256)
    
    # Average over samples
    fine_pred = predictions.mean(axis=1)  # (80, 256)
    
    return coarse_data, fine_gt, fine_pred, timestamp


def visualize_downscaling(coarse_data, fine_gt, fine_pred, timestamp, output_file):
    """
    Visualize downscaling results: coarse input, ground truth, and prediction.
    
    Args:
        coarse_data: (80,) coarse R2B01 values
        fine_gt: (80, 256) ground truth R2B04 field
        fine_pred: (80, 256) predicted R2B04 field
        timestamp: simulation time string
        output_file: path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib not installed")
        return None, None, None
    
    # Reshape for visualization
    # Fine: 80 x 256 = 20,480 cells -> reshape to show detail
    
    # For fine data: show as 80x16 grid (80 coarse cells x 16x16 fine subcells)
    # Each coarse cell has 256 fine cells, arrange as 16x16
    n_coarse = 80
    fine_subcell_size = 16  # sqrt(256) = 16
    fine_grid_rows = 8  # arrange 80 coarse cells in 8x10 grid
    fine_grid_cols = 10
    
    fine_height = fine_grid_rows * fine_subcell_size  # 128
    fine_width = fine_grid_cols * fine_subcell_size   # 160
    
    gt_2d = np.zeros((fine_height, fine_width))
    pred_2d = np.zeros((fine_height, fine_width))
    
    # Fill in each coarse cell's fine data
    for coarse_idx in range(n_coarse):
        row = coarse_idx // fine_grid_cols
        col = coarse_idx % fine_grid_cols
        
        # Get fine cells for this coarse cell (256 values)
        fine_cells_gt = fine_gt[coarse_idx]  # (256,)
        fine_cells_pred = fine_pred[coarse_idx]  # (256,)
        
        # Reshape to 16x16
        fine_patch_gt = fine_cells_gt.reshape(fine_subcell_size, fine_subcell_size)
        fine_patch_pred = fine_cells_pred.reshape(fine_subcell_size, fine_subcell_size)
        
        # Place in grid
        r_start = row * fine_subcell_size
        r_end = r_start + fine_subcell_size
        c_start = col * fine_subcell_size
        c_end = c_start + fine_subcell_size
        
        gt_2d[r_start:r_end, c_start:c_end] = fine_patch_gt
        pred_2d[r_start:r_end, c_start:c_end] = fine_patch_pred
    
    # Compute metrics
    rmse = np.sqrt(((fine_pred - fine_gt) ** 2).mean())
    mae = np.abs(fine_pred - fine_gt).mean()
    bias = (fine_pred - fine_gt).mean()
    
    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Common colormap range
    vmin = min(fine_gt.min(), fine_pred.min())
    vmax = max(fine_gt.max(), fine_pred.max())
    
    # Plot 1: Ground Truth (R2B04)
    im1 = axes[0].imshow(gt_2d, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[0].set_title('Ground Truth (R2B04)\n20480 cells (128×160)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Fine Grid X')
    axes[0].set_ylabel('Fine Grid Y')
    plt.colorbar(im1, ax=axes[0], label='Temperature (K)', fraction=0.046, pad=0.04)
    
    # Plot 2: Prediction (R2B04)
    im2 = axes[1].imshow(pred_2d, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[1].set_title('Diffusion Model Prediction (R2B04)\n20480 cells (128×160)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Fine Grid X')
    axes[1].set_ylabel('Fine Grid Y')
    plt.colorbar(im2, ax=axes[1], label='Temperature (K)', fraction=0.046, pad=0.04)
    
    # Add metrics
    metrics_text = f'RMSE: {rmse:.2e}\nMAE: {mae:.2e}\nBias: {bias:.2e}'
    axes[1].text(0.02, 0.98, metrics_text, transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Diffusion Model Downscaling Test at {timestamp}',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()
    
    return rmse, mae, bias


def main():
    parser = argparse.ArgumentParser(description="Test diffusion model downscaling on ICON data")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        required=True,
        help='Path to test data directory'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of diffusion samples to generate per test case (default: 10)'
    )
    parser.add_argument(
        '--max-tests',
        type=int,
        default=5,
        help='Maximum number of test samples to process (default: 5)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on (default: cpu)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"❌ Error: Test directory not found: {args.test_dir}")
        return
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  Warning: CUDA requested but not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    model, checkpoint = load_model(args.checkpoint, args.device)
    
    # Get test files
    test_files = sorted(glob.glob(os.path.join(args.test_dir, "test_sample_*.npz")))
    print(f"\nFound {len(test_files)} test files")
    
    if not test_files:
        print(f"❌ Error: No test files found in {args.test_dir}")
        return
    
    # Select subset of test files (evenly spaced)
    n_tests = min(args.max_tests, len(test_files))
    indices = np.linspace(0, len(test_files)-1, n_tests, dtype=int)
    test_files = [test_files[i] for i in indices]
    
    print(f"\nTesting on {n_tests} samples...")
    print("="*60)
    
    all_metrics = []
    
    for i, test_file in enumerate(test_files, 1):
        print(f"\nProcessing test {i}/{n_tests}: {os.path.basename(test_file)}")
        
        # Generate predictions
        coarse_data, fine_gt, fine_pred, timestamp = generate_predictions(
            model, test_file, args.device, args.n_samples
        )
        
        # Compute metrics
        rmse = np.sqrt(((fine_pred - fine_gt) ** 2).mean())
        mae = np.abs(fine_pred - fine_gt).mean()
        bias = (fine_pred - fine_gt).mean()
        
        print(f"  RMSE: {rmse:.6e}")
        print(f"  MAE:  {mae:.6e}")
        print(f"  Bias: {bias:.6e}")
        
        all_metrics.append({'rmse': rmse, 'mae': mae, 'bias': bias})
        
        # Visualize if requested
        if args.visualize:
            output_file = f"downscaling_test_{timestamp.replace(':', '-').replace(' ', '_')}.png"
            visualize_downscaling(coarse_data, fine_gt, fine_pred, timestamp, output_file)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    rmses = [m['rmse'] for m in all_metrics]
    maes = [m['mae'] for m in all_metrics]
    biases = [m['bias'] for m in all_metrics]
    
    print(f"RMSE:  mean={np.mean(rmses):.6e}, std={np.std(rmses):.6e}, min={np.min(rmses):.6e}, max={np.max(rmses):.6e}")
    print(f"MAE:   mean={np.mean(maes):.6e}, std={np.std(maes):.6e}, min={np.min(maes):.6e}, max={np.max(maes):.6e}")
    print(f"Bias:  mean={np.mean(biases):.6e}, std={np.std(biases):.6e}, min={np.min(biases):.6e}, max={np.max(biases):.6e}")
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()
