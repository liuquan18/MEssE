#!/usr/bin/env python3
"""
Plot ICON-ComIn Mini-batch GNN training loss curve

Usage:
    python plot_comin_loss.py --log-dir /scratch/m/m301257/icon_exercise_comin
    python plot_comin_loss.py --slurm-file /work/mh1498/m301257/work/MEssE/experiment/slurm.22285966.out
"""

import argparse
import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def read_loss_from_log_files(log_dir):
    """Read loss values from ComIn log file directory (Mini-batch GNN format)"""
    log_files = sorted(glob.glob(os.path.join(log_dir, "log_*.txt")))
    
    if not log_files:
        print(f"Warning: No log files found in {log_dir}")
        return [], [], []
    
    timestep_losses = []  # Average loss per timestep
    batch_losses = []      # All batch losses
    timestamps = []
    
    for log_file in log_files:
        # Extract timestamp from filename
        match = re.search(r'log_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.txt', log_file)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        else:
            timestamp = None
        
        # Read loss values
        try:
            with open(log_file, 'r') as f:
                losses = [float(line.strip()) for line in f if line.strip()]
                
                if losses:
                    # For mini-batch GNN: each timestep has 8 batch losses
                    # Store average as timestep loss
                    avg_loss = np.mean(losses)
                    timestep_losses.append(avg_loss)
                    batch_losses.extend(losses)
                    
                    if timestamp:
                        timestamps.append(timestamp)
        except Exception as e:
            print(f"Error reading file {log_file}: {e}")
            continue
    
    print(f"Read {len(timestep_losses)} timesteps with {len(batch_losses)} total batch losses from {len(log_files)} log files")
    return timestep_losses, batch_losses, timestamps


def read_loss_from_slurm_output(slurm_file):
    """Read loss values from SLURM output file (Mini-batch GNN format)"""
    timestep_losses = []
    batch_losses = []
    current_batches = []
    
    try:
        with open(slurm_file, 'r') as f:
            for line in f:
                # Match Mini-batch GNN batch loss format: "  0:      Loss: 2.705625e-02"
                batch_match = re.match(r'\s*0:\s+Loss:\s*([\d.e+-]+)', line)
                if batch_match:
                    loss_value = float(batch_match.group(1))
                    current_batches.append(loss_value)
                    batch_losses.append(loss_value)
                
                # Match timestep completion marker: "  0: ✓ Mini-batch GNN training completed"
                # or "  0:   Average loss: 9.572431e-03"
                completion_match = re.match(r'\s*0:\s+Average loss:\s*([\d.e+-]+)', line)
                if completion_match:
                    avg_loss = float(completion_match.group(1))
                    timestep_losses.append(avg_loss)
                    current_batches = []
                elif '✓ Mini-batch GNN training completed' in line and current_batches:
                    # Calculate average if not explicitly provided
                    avg_loss = np.mean(current_batches)
                    timestep_losses.append(avg_loss)
                    current_batches = []
    except Exception as e:
        print(f"Error reading SLURM file {slurm_file}: {e}")
        return [], [], []
    
    print(f"Read {len(timestep_losses)} timesteps with {len(batch_losses)} total batch losses from SLURM output file")
    return timestep_losses, batch_losses, []


def plot_loss(timestep_losses, batch_losses, timestamps=None, output_file='loss_curve.png', 
              title='ComIn Mini-batch GNN Training Loss'):
    """Plot loss curve with both timestep-averaged and batch-level losses"""
    if not timestep_losses and not batch_losses:
        print("Error: No loss data to plot")
        return
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Prepare x-axis data for timestep losses
    if timestamps and len(timestamps) == len(timestep_losses):
        x_timestep = timestamps
        x_label = 'Simulation Time'
    else:
        x_timestep = np.arange(1, len(timestep_losses) + 1)
        x_label = 'Timestep'
    
    # Plot 1: Timestep-averaged losses (linear scale)
    if timestep_losses:
        ax1.plot(x_timestep, timestep_losses, linewidth=1.5, alpha=0.8, 
                marker='o', markersize=3, label='Timestep Avg Loss', color='#2E86AB')
        ax1.set_xlabel(x_label, fontsize=11)
        ax1.set_ylabel('Average Loss per Timestep', fontsize=11)
        ax1.set_title(f'{title} - Timestep Averaged ({len(timestep_losses)} timesteps)', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Add statistics text box
        stats_text = (f'Timesteps: {len(timestep_losses)}\n'
                     f'Min: {np.min(timestep_losses):.6e}\n'
                     f'Max: {np.max(timestep_losses):.6e}\n'
                     f'Mean: {np.mean(timestep_losses):.6e}\n'
                     f'Final: {timestep_losses[-1]:.6e}')
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#A8DADC', alpha=0.7),
                fontsize=9, family='monospace')
    
    # Plot 2: Timestep-averaged losses (logarithmic scale)
    if timestep_losses:
        ax2.semilogy(x_timestep, timestep_losses, linewidth=1.5, alpha=0.8,
                    marker='o', markersize=3, color='#E63946', label='Timestep Avg Loss (log scale)')
        ax2.set_xlabel(x_label, fontsize=11)
        ax2.set_ylabel('Average Loss (log scale)', fontsize=11)
        ax2.set_title(f'{title} - Logarithmic Scale', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(loc='upper right')
    
    # Plot 3: All batch losses
    if batch_losses:
        batch_indices = np.arange(1, len(batch_losses) + 1)
        ax3.plot(batch_indices, batch_losses, linewidth=0.5, alpha=0.6, 
                color='#457B9D', label='Individual Batch Loss')
        
        # If we have timestep info, mark timestep boundaries
        num_batches_per_timestep = len(batch_losses) // len(timestep_losses) if timestep_losses else 8
        if timestep_losses and num_batches_per_timestep > 1:
            for i in range(0, len(batch_losses), num_batches_per_timestep):
                ax3.axvline(x=i+1, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        
        ax3.set_xlabel('Batch Index (Global)', fontsize=11)
        ax3.set_ylabel('Batch Loss', fontsize=11)
        ax3.set_title(f'All Batch Losses ({len(batch_losses)} batches, ~{num_batches_per_timestep} batches/timestep)', 
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')
        
        # Add statistics text box
        batch_stats = (f'Total batches: {len(batch_losses)}\n'
                      f'Min: {np.min(batch_losses):.6e}\n'
                      f'Max: {np.max(batch_losses):.6e}\n'
                      f'Mean: {np.mean(batch_losses):.6e}')
        ax3.text(0.02, 0.98, batch_stats, transform=ax3.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#F1FAEE', alpha=0.7),
                fontsize=9, family='monospace')
    
    # Rotate x-axis labels if using timestamps
    if timestamps and len(timestamps) == len(timestep_losses):
        ax1.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='x', rotation=45)
        fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Image saved to: {output_file}")
    
    # Display statistics
    print(f"\n{'='*60}")
    print(f"📊 Mini-batch GNN Loss Statistics")
    print(f"{'='*60}")
    
    if timestep_losses:
        print(f"\n🔹 Timestep-Averaged Losses:")
        print(f"  Total timesteps: {len(timestep_losses)}")
        print(f"  Minimum: {np.min(timestep_losses):.6e}")
        print(f"  Maximum: {np.max(timestep_losses):.6e}")
        print(f"  Mean: {np.mean(timestep_losses):.6e}")
        print(f"  Final: {timestep_losses[-1]:.6e}")
        print(f"  Std Dev: {np.std(timestep_losses):.6e}")
        
        # Calculate loss reduction
        if len(timestep_losses) > 1:
            initial_loss = np.mean(timestep_losses[:min(10, len(timestep_losses))])
            final_loss = np.mean(timestep_losses[-min(10, len(timestep_losses)):])
            reduction = (initial_loss - final_loss) / initial_loss * 100
            print(f"  Loss reduction: {reduction:.2f}% (first 10 vs last 10 timesteps)")
    
    if batch_losses:
        print(f"\n🔹 Batch-Level Losses:")
        print(f"  Total batches: {len(batch_losses)}")
        print(f"  Minimum: {np.min(batch_losses):.6e}")
        print(f"  Maximum: {np.max(batch_losses):.6e}")
        print(f"  Mean: {np.mean(batch_losses):.6e}")
        print(f"  Std Dev: {np.std(batch_losses):.6e}")
        
        if timestep_losses:
            batches_per_timestep = len(batch_losses) // len(timestep_losses)
            print(f"  Batches per timestep: ~{batches_per_timestep}")
    
    print(f"\n{'='*60}")



def main():
    parser = argparse.ArgumentParser(description='Plot ICON-ComIn Mini-batch GNN training loss curve')
    parser.add_argument('--log-dir', type=str, 
                        default='/scratch/m/m301257/icon_exercise_comin',
                        help='ComIn log file directory')
    parser.add_argument('--slurm-file', type=str,
                        help='SLURM output file path (optional, alternative data source)')
    parser.add_argument('--output', '-o', type=str, default='loss_curve_gnn.png',
                        help='Output image filename (default: loss_curve_gnn.png)')
    parser.add_argument('--title', type=str, default='ICON-ComIn Mini-batch GNN Training',
                        help='Chart title')
    
    args = parser.parse_args()
    
    timestep_losses = []
    batch_losses = []
    timestamps = []
    
    # Priority: SLURM file first (more complete), then log directory
    if args.slurm_file and os.path.isfile(args.slurm_file):
        print(f"📄 Reading from SLURM output file: {args.slurm_file}")
        timestep_losses, batch_losses, _ = read_loss_from_slurm_output(args.slurm_file)
    
    # Try log directory if no data from SLURM or as alternative
    if (not timestep_losses and not batch_losses) and args.log_dir and os.path.isdir(args.log_dir):
        print(f"📂 Reading from log directory: {args.log_dir}")
        timestep_losses, batch_losses, timestamps = read_loss_from_log_files(args.log_dir)
    
    # Plot loss curve
    if timestep_losses or batch_losses:
        plot_loss(timestep_losses, batch_losses, timestamps, args.output, args.title)
    else:
        print("❌ Error: Could not read any loss data")
        print(f"\nPlease check:")
        print(f"  1. SLURM file exists: {args.slurm_file}")
        print(f"  2. Log directory exists: {args.log_dir}")
        print(f"\nExpected GNN format:")
        print(f"  - Batch loss: '  0:      Loss: 2.705625e-02'")
        print(f"  - Average loss: '  0:   Average loss: 9.572431e-03'")


if __name__ == '__main__':
    main()
