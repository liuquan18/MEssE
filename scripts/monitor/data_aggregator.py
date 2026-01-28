#!/usr/bin/env python3
"""
Data aggregation module for ICON ML training losses.
Reads loss data from log files and TensorBoard event files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import threading


class LossDataAggregator:
    """Aggregates and manages loss data from ICON ML training."""
    
    def __init__(self, log_dir: Path):
        """
        Initialize the data aggregator.
        
        Args:
            log_dir: Directory containing log files and TensorBoard runs
        """
        self.log_dir = Path(log_dir)
        self.lock = threading.Lock()
        
        # Cached data
        self.loss_data = pd.DataFrame()
        self.summary_data = {}
        self.timesteps = []
        self.total_batches = 0
        self.last_update = None
        
        # TensorBoard data cache
        self.tb_data = {}
        
        # Initial data load
        self.update_data()
    
    def update_data(self):
        """Update cached data from files."""
        with self.lock:
            try:
                self._load_log_files()
                self._load_summary_files()
                self._load_tensorboard_data()
                self.last_update = datetime.now()
            except Exception as e:
                print(f"Error updating data: {e}")
    
    def _load_log_files(self):
        """Load loss values from log files."""
        if not self.log_dir.exists():
            print(f"Warning: Log directory does not exist: {self.log_dir}")
            return
        
        log_files = sorted(self.log_dir.glob("log_*.txt"))
        
        if not log_files:
            print(f"No log files found in {self.log_dir}")
            return
        
        data = []
        
        for log_file in log_files:
            try:
                # Extract timestamp from filename
                filename = log_file.stem
                timestamp_str = filename.replace("log_", "")
                timestamp = pd.to_datetime(timestamp_str)
                
                # Read loss values
                with open(log_file, "r") as f:
                    losses = [float(line.strip()) for line in f if line.strip()]
                
                # Store each batch loss
                for batch_idx, loss in enumerate(losses):
                    data.append({
                        "timestamp": timestamp,
                        "batch": batch_idx,
                        "loss": loss,
                        "global_step": len(data)
                    })
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
        
        if data:
            self.loss_data = pd.DataFrame(data)
            self.timesteps = sorted(self.loss_data['timestamp'].unique())
            self.total_batches = len(self.loss_data)
        else:
            self.loss_data = pd.DataFrame()
            self.timesteps = []
            self.total_batches = 0
    
    def _load_summary_files(self):
        """Load summary statistics from summary files."""
        if not self.log_dir.exists():
            return
        
        summary_files = sorted(self.log_dir.glob("summary_*.txt"))
        self.summary_data = {}
        
        for summary_file in summary_files:
            try:
                # Extract timestamp from filename
                filename = summary_file.stem
                timestamp_str = filename.replace("summary_", "")
                
                # Read summary content
                with open(summary_file, "r") as f:
                    lines = f.readlines()
                
                summary = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Try to convert to appropriate type
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except (ValueError, AttributeError):
                            pass  # Keep as string
                        
                        summary[key] = value
                
                self.summary_data[timestamp_str] = summary
            except Exception as e:
                print(f"Error reading {summary_file}: {e}")
    
    def _load_tensorboard_data(self):
        """Load data from TensorBoard event files."""
        runs_dir = self.log_dir / 'runs'
        
        if not runs_dir.exists():
            return
        
        try:
            # Try to import tensorboard
            from tensorboard.backend.event_processing import event_accumulator
            
            # Find experiment directories
            exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
            
            for exp_dir in exp_dirs:
                try:
                    ea = event_accumulator.EventAccumulator(str(exp_dir))
                    ea.Reload()
                    
                    # Load scalar tags
                    tags = ea.Tags().get('scalars', [])
                    
                    for tag in tags:
                        events = ea.Scalars(tag)
                        self.tb_data[tag] = [
                            {'step': e.step, 'value': e.value, 'wall_time': e.wall_time}
                            for e in events
                        ]
                except Exception as e:
                    print(f"Error loading TensorBoard data from {exp_dir}: {e}")
        except ImportError:
            # TensorBoard not available
            pass
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current (most recent) loss statistics."""
        with self.lock:
            if self.loss_data.empty:
                return {'error': 'No data available'}
            
            # Get most recent timestep
            latest_timestep = self.timesteps[-1] if self.timesteps else None
            
            if latest_timestep is None:
                return {'error': 'No timesteps available'}
            
            # Get losses for latest timestep
            latest_losses = self.loss_data[
                self.loss_data['timestamp'] == latest_timestep
            ]['loss'].values
            
            return {
                'timestamp': str(latest_timestep),
                'mean_loss': float(np.mean(latest_losses)),
                'std_loss': float(np.std(latest_losses)),
                'min_loss': float(np.min(latest_losses)),
                'max_loss': float(np.max(latest_losses)),
                'num_batches': len(latest_losses),
                'latest_loss': float(latest_losses[-1]) if len(latest_losses) > 0 else None
            }
    
    def get_loss_history(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get historical loss data.
        
        Args:
            limit: Maximum number of data points to return
        """
        with self.lock:
            if self.loss_data.empty:
                return {'timestamps': [], 'losses': [], 'mean_losses': []}
            
            # Get last N data points
            df = self.loss_data.tail(limit)
            
            # Calculate mean loss per timestep
            mean_per_timestep = df.groupby('timestamp')['loss'].agg(['mean', 'std', 'min', 'max'])
            
            return {
                'timestamps': [str(ts) for ts in mean_per_timestep.index],
                'mean_losses': mean_per_timestep['mean'].tolist(),
                'std_losses': mean_per_timestep['std'].tolist(),
                'min_losses': mean_per_timestep['min'].tolist(),
                'max_losses': mean_per_timestep['max'].tolist(),
                'all_losses': df['loss'].tolist(),
                'all_timestamps': [str(ts) for ts in df['timestamp']],
                'batch_indices': df['batch'].tolist()
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistical summary."""
        with self.lock:
            if self.loss_data.empty:
                return {'error': 'No data available'}
            
            losses = self.loss_data['loss'].values
            
            # Calculate percentiles
            percentiles = np.percentile(losses, [25, 50, 75, 90, 95, 99])
            
            # Calculate trend (simple linear fit)
            if len(losses) > 1:
                x = np.arange(len(losses))
                coeffs = np.polyfit(x, losses, 1)
                trend = 'decreasing' if coeffs[0] < 0 else 'increasing'
                trend_slope = float(coeffs[0])
            else:
                trend = 'unknown'
                trend_slope = 0.0
            
            return {
                'total_batches': int(self.total_batches),
                'total_timesteps': len(self.timesteps),
                'overall_mean': float(np.mean(losses)),
                'overall_std': float(np.std(losses)),
                'overall_min': float(np.min(losses)),
                'overall_max': float(np.max(losses)),
                'percentile_25': float(percentiles[0]),
                'percentile_50': float(percentiles[1]),
                'percentile_75': float(percentiles[2]),
                'percentile_90': float(percentiles[3]),
                'percentile_95': float(percentiles[4]),
                'percentile_99': float(percentiles[5]),
                'trend': trend,
                'trend_slope': trend_slope,
                'first_loss': float(losses[0]) if len(losses) > 0 else None,
                'latest_loss': float(losses[-1]) if len(losses) > 0 else None,
                'improvement': float(losses[0] - losses[-1]) if len(losses) > 1 else None
            }
    
    def get_timesteps(self) -> List[str]:
        """Get list of all available timesteps."""
        with self.lock:
            return [str(ts) for ts in self.timesteps]
    
    def get_timestep_data(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """
        Get all data for a specific timestep.
        
        Args:
            timestamp: Timestamp string
        """
        with self.lock:
            if self.loss_data.empty:
                return None
            
            try:
                ts = pd.to_datetime(timestamp)
                data = self.loss_data[self.loss_data['timestamp'] == ts]
                
                if data.empty:
                    return None
                
                losses = data['loss'].values
                
                result = {
                    'timestamp': timestamp,
                    'losses': losses.tolist(),
                    'batch_indices': data['batch'].tolist(),
                    'mean_loss': float(np.mean(losses)),
                    'std_loss': float(np.std(losses)),
                    'min_loss': float(np.min(losses)),
                    'max_loss': float(np.max(losses)),
                    'num_batches': len(losses)
                }
                
                # Add summary data if available
                if timestamp in self.summary_data:
                    result['summary'] = self.summary_data[timestamp]
                
                return result
            except Exception as e:
                print(f"Error getting timestep data: {e}")
                return None
    
    def has_tensorboard_data(self) -> bool:
        """Check if TensorBoard data is available."""
        with self.lock:
            return len(self.tb_data) > 0
    
    def get_tensorboard_scalars(self, tag: str) -> Dict[str, Any]:
        """
        Get scalar data from TensorBoard for a specific tag.
        
        Args:
            tag: TensorBoard scalar tag (e.g., 'Loss/batch')
        """
        with self.lock:
            if tag not in self.tb_data:
                return {'error': f'Tag {tag} not found', 'available_tags': list(self.tb_data.keys())}
            
            data = self.tb_data[tag]
            
            return {
                'tag': tag,
                'steps': [d['step'] for d in data],
                'values': [d['value'] for d in data],
                'wall_times': [d['wall_time'] for d in data],
                'count': len(data)
            }
    
    def get_tensorboard_tags(self) -> List[str]:
        """Get all available TensorBoard tags."""
        with self.lock:
            return list(self.tb_data.keys())
