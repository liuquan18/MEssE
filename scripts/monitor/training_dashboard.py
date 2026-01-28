#!/usr/bin/env python3
"""
Real-time Training Dashboard
Interactive monitoring of ICON model training progress with auto-refresh.
"""

import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import getpass
from datetime import datetime

# Get username
user = getpass.getuser()

# Set paths
LOG_DIR = f"/scratch/{user[0]}/{user}/icon_exercise_comin"

# Dashboard configuration
REFRESH_INTERVAL = 5  # seconds
MAX_POINTS = 1000  # Maximum points to display for performance


class TrainingDashboard:
    """Real-time training monitoring dashboard."""
    
    def __init__(self, log_dir, refresh_interval=5):
        self.log_dir = log_dir
        self.refresh_interval = refresh_interval
        self.last_update = None
        
        # Initialize data storage
        self.losses = []
        self.batches = []
        self.timesteps = []
        self.summary_data = []
        
        # Setup plot
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('ICON Training Monitor - Real-time Dashboard', 
                         fontsize=16, fontweight='bold')
        
        # Create subplots
        self.ax1 = plt.subplot(2, 3, 1)  # Loss vs batch
        self.ax2 = plt.subplot(2, 3, 2)  # Loss vs time
        self.ax3 = plt.subplot(2, 3, 3)  # Loss distribution
        self.ax4 = plt.subplot(2, 3, 4)  # Mean loss per timestep
        self.ax5 = plt.subplot(2, 3, 5)  # Statistics text
        self.ax6 = plt.subplot(2, 3, 6)  # Recent loss (last 100 batches)
        
        # Remove axes from statistics panel
        self.ax5.axis('off')
        
        print(f"Dashboard initialized. Monitoring: {log_dir}")
        print(f"Refresh interval: {refresh_interval} seconds")
        print("Close the window to exit.\n")
    
    def load_data(self):
        """Load and update data from log files."""
        # Parse log files
        log_files = sorted(glob.glob(os.path.join(self.log_dir, "log_*.txt")))
        
        if not log_files:
            return False
        
        # Check if there are new files
        latest_file_time = max([os.path.getmtime(f) for f in log_files])
        if self.last_update and latest_file_time <= self.last_update:
            return False  # No new data
        
        self.last_update = latest_file_time
        
        # Reset data
        self.losses = []
        self.timesteps = []
        
        for log_file in log_files:
            filename = os.path.basename(log_file)
            timestamp_str = filename.replace("log_", "").replace(".txt", "")
            
            try:
                timestamp = pd.to_datetime(timestamp_str)
            except:
                continue
            
            with open(log_file, 'r') as f:
                losses = [float(line.strip()) for line in f if line.strip()]
            
            self.losses.extend(losses)
            self.timesteps.extend([timestamp] * len(losses))
        
        self.batches = list(range(len(self.losses)))
        
        # Parse summary files
        summary_files = sorted(glob.glob(os.path.join(self.log_dir, "summary_*.txt")))
        self.summary_data = []
        
        for summary_file in summary_files:
            summary_dict = {}
            with open(summary_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        summary_dict[key.strip()] = value.strip()
            
            try:
                self.summary_data.append({
                    'timestep': pd.to_datetime(summary_dict['Timestep']),
                    'mean_loss': float(summary_dict['Mean Loss']),
                    'std_loss': float(summary_dict['Std Loss']),
                    'min_loss': float(summary_dict['Min Loss']),
                    'max_loss': float(summary_dict['Max Loss'])
                })
            except (KeyError, ValueError):
                continue
        
        return True
    
    def update_plots(self, frame):
        """Update all plots with latest data."""
        # Try to load new data
        has_new_data = self.load_data()
        
        if not self.losses:
            return
        
        # Limit data points for performance
        if len(self.losses) > MAX_POINTS:
            step = len(self.losses) // MAX_POINTS
            plot_batches = self.batches[::step]
            plot_losses = [self.losses[i] for i in range(0, len(self.losses), step)]
            plot_timesteps = [self.timesteps[i] for i in range(0, len(self.timesteps), step)]
        else:
            plot_batches = self.batches
            plot_losses = self.losses
            plot_timesteps = self.timesteps
        
        # Clear all axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.ax6.clear()
        
        # Plot 1: Loss vs Batch
        self.ax1.plot(plot_batches, plot_losses, linewidth=1, alpha=0.7)
        self.ax1.set_xlabel('Batch Number')
        self.ax1.set_ylabel('Loss (MSE)')
        self.ax1.set_title('Loss vs Batch')
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss vs Time
        self.ax2.plot(plot_timesteps, plot_losses, linewidth=1, alpha=0.7)
        self.ax2.set_xlabel('Simulation Time')
        self.ax2.set_ylabel('Loss (MSE)')
        self.ax2.set_title('Loss vs Time')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Loss Distribution
        self.ax3.hist(self.losses, bins=30, alpha=0.7, edgecolor='black')
        self.ax3.axvline(np.mean(self.losses), color='r', linestyle='--', 
                        linewidth=2, label=f'Mean: {np.mean(self.losses):.6f}')
        self.ax3.set_xlabel('Loss (MSE)')
        self.ax3.set_ylabel('Frequency')
        self.ax3.set_title('Loss Distribution')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
        # Plot 4: Mean loss per timestep (if available)
        if self.summary_data:
            summary_df = pd.DataFrame(self.summary_data)
            self.ax4.errorbar(summary_df['timestep'], summary_df['mean_loss'],
                            yerr=summary_df['std_loss'], fmt='o-', capsize=5, alpha=0.7)
            self.ax4.set_xlabel('Simulation Time')
            self.ax4.set_ylabel('Mean Loss')
            self.ax4.set_title('Mean Loss per Timestep')
            self.ax4.grid(True, alpha=0.3)
            self.ax4.tick_params(axis='x', rotation=45)
        else:
            self.ax4.text(0.5, 0.5, 'Summary data not available',
                         ha='center', va='center', transform=self.ax4.transAxes)
            self.ax4.set_title('Mean Loss per Timestep')
        
        # Plot 5: Statistics text
        self.ax5.axis('off')
        stats_text = self._generate_stats_text()
        self.ax5.text(0.1, 0.95, stats_text, transform=self.ax5.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Plot 6: Recent losses (last 100 batches)
        recent_n = min(100, len(self.losses))
        if recent_n > 0:
            recent_batches = self.batches[-recent_n:]
            recent_losses = self.losses[-recent_n:]
            self.ax6.plot(recent_batches, recent_losses, linewidth=1.5, alpha=0.8)
            self.ax6.set_xlabel('Batch Number')
            self.ax6.set_ylabel('Loss (MSE)')
            self.ax6.set_title(f'Recent Loss (last {recent_n} batches)')
            self.ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def _generate_stats_text(self):
        """Generate statistics text for display."""
        if not self.losses:
            return "Waiting for training data..."
        
        losses_array = np.array(self.losses)
        
        stats = [
            "╔═══════════════════════════════╗",
            "║   TRAINING STATISTICS         ║",
            "╚═══════════════════════════════╝",
            "",
            f"Total Batches:  {len(self.losses):,}",
            "",
            "Loss Statistics:",
            f"  Mean:    {losses_array.mean():.8f}",
            f"  Median:  {np.median(losses_array):.8f}",
            f"  Std Dev: {losses_array.std():.8f}",
            f"  Min:     {losses_array.min():.8f}",
            f"  Max:     {losses_array.max():.8f}",
            "",
        ]
        
        if self.summary_data:
            stats.extend([
                f"Timesteps: {len(self.summary_data)}",
                f"First: {self.summary_data[0]['timestep'].strftime('%Y-%m-%d %H:%M')}",
                f"Last:  {self.summary_data[-1]['timestep'].strftime('%Y-%m-%d %H:%M')}",
                "",
            ])
        
        # Recent trend
        if len(self.losses) > 20:
            recent_mean = np.mean(self.losses[-20:])
            older_mean = np.mean(self.losses[-40:-20]) if len(self.losses) > 40 else np.mean(self.losses[:20])
            trend = "↓ Decreasing" if recent_mean < older_mean else "↑ Increasing"
            change = abs(recent_mean - older_mean)
            stats.extend([
                "Recent Trend:",
                f"  {trend}",
                f"  Change: {change:.8f}",
                "",
            ])
        
        stats.extend([
            f"Last Update:",
            f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ])
        
        return '\n'.join(stats)
    
    def run(self):
        """Start the dashboard."""
        # Load initial data
        self.load_data()
        
        # Create animation
        anim = FuncAnimation(self.fig, self.update_plots, 
                           interval=self.refresh_interval * 1000,
                           cache_frame_data=False)
        
        plt.show()


def main():
    """Main function."""
    print("="*60)
    print("ICON Training Real-time Dashboard")
    print("="*60)
    print(f"Log directory: {LOG_DIR}\n")
    
    # Check if log directory exists
    if not os.path.exists(LOG_DIR):
        print(f"Error: Log directory does not exist: {LOG_DIR}")
        print("Please run the ICON model with the plugin first.")
        sys.exit(1)
    
    # Create and run dashboard
    dashboard = TrainingDashboard(LOG_DIR, refresh_interval=REFRESH_INTERVAL)
    
    try:
        dashboard.run()
    except KeyboardInterrupt:
        print("\nDashboard closed by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
