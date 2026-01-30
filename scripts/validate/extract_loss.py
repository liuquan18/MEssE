#!/usr/bin/env python3
"""
Extract, save, and visualize loss function values from training log files.
This script reads all training session logs, extracts loss values, saves them to files,
and generates comprehensive visualizations.

Usage:
    python extract_loss.py                    # Extract and print summary
    python extract_loss.py --plot             # Extract, save, and plot
    python extract_loss.py --log-dir /path    # Extract from custom directory
    python extract_loss.py --no-save          # Only plot, don't save JSON/CSV
"""

import glob
import re
import os
import argparse
import getpass
from datetime import datetime
import json


def extract_loss_from_logs(log_dir):
    """
    Extract loss values from all training session log files.
    
    Args:
        log_dir: Directory containing log_session_*.txt files
        
    Returns:
        loss_data: List of dicts with format:
            {
                'session': int,           # Training session number
                'timestamp': str,         # ISO format timestamp
                'epoch_losses': [float],  # List of loss values per epoch
                'final_loss': float,      # Last epoch loss
                'log_file': str          # Source file path
            }
    """
    # Find all log files matching pattern: log_session_XXXX_YYYY-MM-DDTHH-MM-SS.txt
    log_pattern = os.path.join(log_dir, "log_session_*.txt")
    log_files = sorted(glob.glob(log_pattern))
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        print(f"Looking for pattern: {log_pattern}")
        return []
    
    print(f"Found {len(log_files)} log files")
    
    loss_data = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Parse log file
            session_num = None
            timestamp = None
            epoch_losses = []
            
            for line in lines:
                line = line.strip()
                
                # Extract session number: "training_session: 123"
                if line.startswith("training_session:"):
                    session_num = int(line.split(":")[-1].strip())
                
                # Extract timestamp: "simulation_time: 2024-01-15T12-30-00"
                elif line.startswith("simulation_time:"):
                    timestamp = line.split(":", 1)[-1].strip()
                
                # Extract epoch loss: "epoch_1: 0.123456"
                elif line.startswith("epoch_"):
                    loss_value = float(line.split(":")[-1].strip())
                    epoch_losses.append(loss_value)
            
            # Validate extracted data
            if session_num is not None and timestamp is not None and epoch_losses:
                loss_data.append({
                    'session': session_num,
                    'timestamp': timestamp,
                    'epoch_losses': epoch_losses,
                    'final_loss': epoch_losses[-1],
                    'log_file': log_file
                })
            else:
                print(f"Warning: Incomplete data in {log_file}")
                
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
            continue
    
    # Sort by session number
    loss_data.sort(key=lambda x: x['session'])
    
    return loss_data


def save_loss_to_json(loss_data, output_file):
    """
    Save loss data to JSON file for easy analysis.
    
    Args:
        loss_data: List of loss dicts from extract_loss_from_logs()
        output_file: Path to output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(loss_data, f, indent=2)
    print(f"Saved loss data to: {output_file}")


def save_loss_to_csv(loss_data, output_file):
    """
    Save loss data to CSV file for spreadsheet analysis.
    
    Args:
        loss_data: List of loss dicts from extract_loss_from_logs()
        output_file: Path to output CSV file
    """
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['session', 'timestamp', 'epoch', 'loss'])
        
        # Data rows
        for entry in loss_data:
            for epoch_idx, loss_val in enumerate(entry['epoch_losses'], start=1):
                writer.writerow([
                    entry['session'],
                    entry['timestamp'],
                    epoch_idx,
                    f"{loss_val:.6f}"
                ])
    
    print(f"Saved loss data to: {output_file}")


def print_loss_summary(loss_data):
    """
    Print a summary of loss values to console.
    
    Args:
        loss_data: List of loss dicts from extract_loss_from_logs()
    """
    if not loss_data:
        print("No loss data to summarize")
        return
    
    print("\n" + "="*80)
    print("LOSS FUNCTION SUMMARY")
    print("="*80)
    print(f"Total training sessions: {len(loss_data)}")
    print(f"First session: {loss_data[0]['session']} at {loss_data[0]['timestamp']}")
    print(f"Last session:  {loss_data[-1]['session']} at {loss_data[-1]['timestamp']}")
    print("-"*80)
    
    # Print each session
    for entry in loss_data:
        print(f"Session {entry['session']:4d} | {entry['timestamp']} | "
              f"Epochs: {len(entry['epoch_losses'])} | "
              f"Final loss: {entry['final_loss']:.6f}")
        
        # Show all epoch losses if multiple epochs
        if len(entry['epoch_losses']) > 1:
            epoch_str = " -> ".join([f"{loss:.6f}" for loss in entry['epoch_losses']])
            print(f"              Loss progression: {epoch_str}")
    
    print("-"*80)
    
    # Statistics
    final_losses = [entry['final_loss'] for entry in loss_data]
    print(f"Loss statistics (final epoch only):")
    print(f"  Min:    {min(final_losses):.6f}")
    print(f"  Max:    {max(final_losses):.6f}")
    print(f"  Mean:   {sum(final_losses)/len(final_losses):.6f}")
    print(f"  Latest: {final_losses[-1]:.6f}")
    print("="*80 + "\n")


def plot_loss_evolution(loss_data, output_file=None, show_plot=False):
    """
    Generate plot showing loss evolution over model time.
    
    Args:
        loss_data: List of loss dicts from extract_loss_from_logs()
        output_file: Path to save plot (default: loss_evolution.png)
        show_plot: Whether to display plot interactively (default: False)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: matplotlib not installed. Cannot generate plot.")
        print("Install with: pip install matplotlib")
        return
    
    if not loss_data:
        print("No loss data to plot")
        return
    
    # Extract data for plotting
    sessions = [entry['session'] for entry in loss_data]
    final_losses = [entry['final_loss'] for entry in loss_data]
    
    # Parse timestamps and calculate hours from start
    timestamps = []
    for entry in loss_data:
        time_str = entry['timestamp']
        try:
            # Try ISO format with hyphens (1979-01-03T12-00-00)
            timestamp = datetime.strptime(time_str, "%Y-%m-%dT%H-%M-%S")
        except ValueError:
            try:
                # Try format with colons (1979-01-03 12:00:00)
                timestamp = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                print(f"Warning: Could not parse timestamp {time_str}")
                timestamp = None
        timestamps.append(timestamp)
    
    # Calculate relative time in hours
    if timestamps[0]:
        start_time = timestamps[0]
        hours = [(t - start_time).total_seconds() / 3600 if t else 0 for t in timestamps]
    else:
        hours = list(range(len(sessions)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot loss vs model time
    ax.plot(hours, final_losses, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    
    ax.set_xlabel('Model Time', fontsize=14)
    ax.set_ylabel('Loss (MSE)', fontsize=14)
    ax.set_title('Diffusion Model Training Loss Evolution', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics box
    latest_loss = final_losses[-1]
    min_loss = min(final_losses)
    textstr = f'Latest: {latest_loss:.6f}\nMin: {min_loss:.6f}\nSessions: {len(sessions)}'
    if hours[-1] > 0:
        textstr += f'\nDuration: {hours[-1]:.1f}h'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved plot to: {output_file}")
    
    # Show plot interactively
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Print training summary
    print(f"\nTraining Summary:")
    print(f"  Sessions: {len(sessions)}")
    if hours[-1] > 0:
        print(f"  Duration: {hours[-1]:.1f} hours")
    print(f"  Initial loss: {final_losses[0]:.6f}")
    print(f"  Final loss: {final_losses[-1]:.6f}")
    print(f"  Best loss: {min(final_losses):.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract loss function values from diffusion model training logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract and print summary only
  python extract_loss.py
  
  # Extract, save to files, and generate plot
  python extract_loss.py --plot
  
  # Extract with custom output location
  python extract_loss.py --plot --output-dir ./results
  
  # Show plot interactively
  python extract_loss.py --plot --show
  
  # Extract from custom log directory
  python extract_loss.py --log-dir /path/to/logs --plot
        """
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Directory containing log files (default: /scratch/<user>/icon_exercise_comin/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory for output files (default: current directory)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default='loss_history.json',
        help='Output JSON file name (default: loss_history.json)'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default='loss_history.csv',
        help='Output CSV file name (default: loss_history.csv)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate loss evolution plot'
    )
    parser.add_argument(
        '--plot-file',
        type=str,
        default='loss_evolution.png',
        help='Plot output file name (default: loss_evolution.png)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plot interactively (requires --plot)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving JSON/CSV files, only generate plot'
    )
    
    args = parser.parse_args()
    
    # Determine log directory
    if args.log_dir is None:
        user = getpass.getuser()
        args.log_dir = f"/scratch/{user[0]}/{user}/icon_exercise_comin"
    
    print(f"Reading logs from: {args.log_dir}")
    
    # Extract loss data
    loss_data = extract_loss_from_logs(args.log_dir)
    
    if not loss_data:
        print("No loss data found. Exiting.")
        return
    
    # Print summary
    print_loss_summary(loss_data)
    
    # Prepare output paths
    output_json = os.path.join(args.output_dir, args.output_json)
    output_csv = os.path.join(args.output_dir, args.output_csv)
    output_plot = os.path.join(args.output_dir, args.plot_file)
    
    # Save to JSON and CSV (unless --no-save)
    if not args.no_save:
        save_loss_to_json(loss_data, output_json)
        save_loss_to_csv(loss_data, output_csv)
    
    # Generate plot if requested
    if args.plot:
        plot_loss_evolution(loss_data, output_plot, show_plot=args.show)
    elif args.show:
        print("Warning: --show requires --plot flag. Use: python extract_loss.py --plot --show")


if __name__ == "__main__":
    main()
