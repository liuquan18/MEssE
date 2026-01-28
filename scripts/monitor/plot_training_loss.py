#!/usr/bin/env python3
"""
Training Loss Visualization Script
Parses log files from ICON model training and generates loss plots.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import getpass

# Get username
user = getpass.getuser()

# Set paths
LOG_DIR = f"/scratch/{user[0]}/{user}/icon_exercise_comin"
OUTPUT_DIR = os.path.join(LOG_DIR, "plots")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_log_files():
    """Parse all log files and return aggregated data."""
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "log_*.txt")))
    
    if not log_files:
        print(f"No log files found in {LOG_DIR}")
        return None
    
    print(f"Found {len(log_files)} log files")
    
    all_losses = []
    timesteps = []
    
    for log_file in log_files:
        # Extract timestamp from filename
        filename = os.path.basename(log_file)
        timestamp_str = filename.replace("log_", "").replace(".txt", "")
        
        try:
            timestamp = pd.to_datetime(timestamp_str)
        except:
            print(f"Warning: Could not parse timestamp from {filename}")
            continue
        
        # Read losses from file
        with open(log_file, 'r') as f:
            losses = [float(line.strip()) for line in f if line.strip()]
        
        all_losses.extend(losses)
        timesteps.extend([timestamp] * len(losses))
    
    if not all_losses:
        print("No loss data found in log files")
        return None
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestep': timesteps,
        'loss': all_losses,
        'batch': range(len(all_losses))
    })
    
    return df


def parse_summary_files():
    """Parse summary files for timestep-level statistics."""
    summary_files = sorted(glob.glob(os.path.join(LOG_DIR, "summary_*.txt")))
    
    if not summary_files:
        print("No summary files found")
        return None
    
    print(f"Found {len(summary_files)} summary files")
    
    data = []
    
    for summary_file in summary_files:
        summary_dict = {}
        with open(summary_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    summary_dict[key.strip()] = value.strip()
        
        try:
            data.append({
                'timestep': pd.to_datetime(summary_dict['Timestep']),
                'global_step': int(summary_dict['Global Step']),
                'num_batches': int(summary_dict['Number of Batches']),
                'mean_loss': float(summary_dict['Mean Loss']),
                'std_loss': float(summary_dict['Std Loss']),
                'min_loss': float(summary_dict['Min Loss']),
                'max_loss': float(summary_dict['Max Loss'])
            })
        except KeyError as e:
            print(f"Warning: Missing key {e} in {summary_file}")
            continue
    
    if not data:
        return None
    
    return pd.DataFrame(data)


def plot_loss_curves(df, output_dir):
    """Generate comprehensive loss visualization plots."""
    
    # Plot 1: Loss vs Batch (all batches)
    plt.figure(figsize=(14, 6))
    plt.plot(df['batch'], df['loss'], alpha=0.7, linewidth=0.5)
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Loss vs Batch Number', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_vs_batch.png'), dpi=300)
    print(f"Saved: {os.path.join(output_dir, 'loss_vs_batch.png')}")
    plt.close()
    
    # Plot 2: Loss vs Time (timestep)
    plt.figure(figsize=(14, 6))
    plt.plot(df['timestep'], df['loss'], alpha=0.7, linewidth=0.5)
    plt.xlabel('Simulation Time', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Loss vs Simulation Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_vs_time.png'), dpi=300)
    print(f"Saved: {os.path.join(output_dir, 'loss_vs_time.png')}")
    plt.close()
    
    # Plot 3: Loss distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['loss'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Loss (MSE)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Loss Distribution', fontsize=14, fontweight='bold')
    plt.axvline(df['loss'].mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {df["loss"].mean():.6f}')
    plt.axvline(df['loss'].median(), color='g', linestyle='--', linewidth=2, label=f'Median: {df["loss"].median():.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_distribution.png'), dpi=300)
    print(f"Saved: {os.path.join(output_dir, 'loss_distribution.png')}")
    plt.close()
    
    # Plot 4: Moving average (smoothed loss)
    window_size = min(100, len(df) // 10)
    if window_size > 1:
        plt.figure(figsize=(14, 6))
        plt.plot(df['batch'], df['loss'], alpha=0.3, linewidth=0.5, label='Raw Loss')
        moving_avg = df['loss'].rolling(window=window_size, center=True).mean()
        plt.plot(df['batch'], moving_avg, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
        plt.xlabel('Batch Number', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Training Loss with Moving Average', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_smoothed.png'), dpi=300)
        print(f"Saved: {os.path.join(output_dir, 'loss_smoothed.png')}")
        plt.close()


def plot_timestep_statistics(summary_df, output_dir):
    """Plot timestep-level statistics."""
    
    if summary_df is None or len(summary_df) == 0:
        print("No summary data available for plotting")
        return
    
    # Plot 1: Mean loss per timestep with error bars
    plt.figure(figsize=(14, 6))
    plt.errorbar(summary_df['timestep'], summary_df['mean_loss'], 
                 yerr=summary_df['std_loss'], fmt='o-', capsize=5, alpha=0.7)
    plt.xlabel('Simulation Time', fontsize=12)
    plt.ylabel('Mean Loss (MSE)', fontsize=12)
    plt.title('Mean Loss per Timestep (with Std Dev)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_loss_per_timestep.png'), dpi=300)
    print(f"Saved: {os.path.join(output_dir, 'mean_loss_per_timestep.png')}")
    plt.close()
    
    # Plot 2: Min/Max loss range per timestep
    plt.figure(figsize=(14, 6))
    plt.fill_between(summary_df['timestep'], summary_df['min_loss'], 
                     summary_df['max_loss'], alpha=0.3, label='Min-Max Range')
    plt.plot(summary_df['timestep'], summary_df['mean_loss'], 
             'o-', linewidth=2, label='Mean Loss')
    plt.xlabel('Simulation Time', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Loss Range per Timestep', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_range_per_timestep.png'), dpi=300)
    print(f"Saved: {os.path.join(output_dir, 'loss_range_per_timestep.png')}")
    plt.close()
    
    # Plot 3: Number of batches per timestep
    plt.figure(figsize=(14, 6))
    plt.bar(summary_df['timestep'], summary_df['num_batches'], alpha=0.7)
    plt.xlabel('Simulation Time', fontsize=12)
    plt.ylabel('Number of Batches', fontsize=12)
    plt.title('Training Batches per Timestep', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batches_per_timestep.png'), dpi=300)
    print(f"Saved: {os.path.join(output_dir, 'batches_per_timestep.png')}")
    plt.close()


def print_statistics(df, summary_df):
    """Print training statistics."""
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    
    print(f"\nTotal batches processed: {len(df)}")
    print(f"Loss statistics:")
    print(f"  Mean:   {df['loss'].mean():.8f}")
    print(f"  Median: {df['loss'].median():.8f}")
    print(f"  Std:    {df['loss'].std():.8f}")
    print(f"  Min:    {df['loss'].min():.8f}")
    print(f"  Max:    {df['loss'].max():.8f}")
    
    if summary_df is not None and len(summary_df) > 0:
        print(f"\nTimesteps processed: {len(summary_df)}")
        print(f"  First: {summary_df['timestep'].iloc[0]}")
        print(f"  Last:  {summary_df['timestep'].iloc[-1]}")
        
        # Check for convergence
        if len(summary_df) > 1:
            first_half_mean = summary_df['mean_loss'].iloc[:len(summary_df)//2].mean()
            second_half_mean = summary_df['mean_loss'].iloc[len(summary_df)//2:].mean()
            improvement = ((first_half_mean - second_half_mean) / first_half_mean) * 100
            print(f"\nConvergence analysis:")
            print(f"  First half mean loss:  {first_half_mean:.8f}")
            print(f"  Second half mean loss: {second_half_mean:.8f}")
            print(f"  Improvement:           {improvement:.2f}%")
    
    print("="*60 + "\n")


def main():
    """Main function."""
    print("="*60)
    print("ICON Training Loss Visualization")
    print("="*60)
    print(f"Log directory: {LOG_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Parse data
    print("Parsing log files...")
    df = parse_log_files()
    
    if df is None:
        print("Error: No data to plot")
        sys.exit(1)
    
    print("Parsing summary files...")
    summary_df = parse_summary_files()
    
    # Print statistics
    print_statistics(df, summary_df)
    
    # Generate plots
    print("Generating plots...")
    plot_loss_curves(df, OUTPUT_DIR)
    
    if summary_df is not None:
        plot_timestep_statistics(summary_df, OUTPUT_DIR)
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
