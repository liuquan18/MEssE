# Monitor System Updates

## Overview
The monitoring system has been updated to support the enhanced features in the updated `gnn_trainer_gpu.py`, including:

1. **GNN Mini-batch Training**: Monitors Graph Neural Network training with spatial batching
2. **24-Hour Training Delay**: Training begins only after 24 hours of simulation elapsed time
3. **Smart Checkpoint Management**: Model loads from checkpoint if exists, otherwise initializes with random weights
4. **Mini-batch Training Metrics**: Tracks batch counts per timestep for mini-batch GNN training
5. **Enhanced Loss Statistics**: Includes min, max, and average loss values
6. **Real-time Status Writing**: The plugin now directly writes monitoring status JSON

## Key Changes

### 1. gnn_trainer_gpu.py
- Training starts only after 24 hours of simulation elapsed time
- Model initialization:
  - First training session: Random weight initialization
  - Subsequent sessions: Loads from checkpoint
- Checkpoint logic based on file existence (not timestep)
- Writes comprehensive status including:
  - Model type (GNN Mini-batch)
  - Training enabled status
  - Elapsed hours tracking
  - Number of nodes and batches
  - Training statistics (avg, min, max loss)
  
### 2. generate_status.py
- Updated to use user-specific scratch directory (works for any user)
- Enhanced timestamp parsing (supports both old and new formats)
- Model type detection from checkpoint files
- Additional metrics:
  - `model_type`: "GNN (Mini-batch)"
  - `training_enabled`: Boolean indicating if 24-hour threshold passed
  - `elapsed_hours`: Hours since simulation start
  - `batches_per_timestep`: Number of batches processed per output
  - `max_loss`: Maximum loss value
- Better error handling

### 3. monitor.html
- Updated to display model type
- Shows batches per timestep
- Displays max loss in addition to min and average
- Uses scientific notation for loss values (better for very small numbers)

### 4. monitor_server.py
- No changes needed (already compatible)

## Usage

### Running the Monitor

1. **Start the simulation** (if not already running):
   ```bash
   cd /work/mh0033/m300883/Project_week/MEssE/scripts/run_icon
   ./run_icon_LAM.sh
   ```

2. **Start the monitoring server**:
   ```bash
   cd /work/mh0033/m300883/Project_week/MEssE/scripts/monitor
   ./start_monitor.sh
   ```

3. **Access the web interface**:
   - Open your browser to: `http://localhost:5001`
   - Or if on a remote server, use SSH port forwarding:
     ```bash
     ssh -L 5001:localhost:5001 user@host
     ```

### Manual Status Generation

If you need to manually generate the status file:
```bash
cd /work/mh0033/m300883/Project_week/MEssE/scripts/monitor
source /work/mh0033/m300883/Project_week/MEssE/build/messe_env/py_env/bin/activate
python generate_status.py
```

## File Locations

- **Log files**: `/scratch/{first_letter_of_user}/{user}/icon_exercise_comin/log_*.txt`
- **Checkpoint files**: `/scratch/{first_letter_of_user}/{user}/icon_exercise_comin/net_*.pth`
- **Status file**: `/scratch/{first_letter_of_user}/{user}/icon_exercise_comin/monitor_status.json`

## New Features in the Web Interface

### Simulation Panel
- Shows all simulation parameters
- Real-time updates every 2 seconds
- Displays elapsed time and output count

### Training Panel
- **Model Type**: Displays "GNN (Mini-batch)"
- **Training Status**: Shows if training is enabled (elapsed > 24 hours)
- **Elapsed Time**: Hours since simulation start
- **Current Loss**: Latest loss value in scientific notation
- **Total Batches**: Cumulative batch count
- **Batches/Timestep**: Number of batches per output timestep
- **Loss Statistics**: Average, minimum, and maximum
- **Learning Rate**: 0.001 (fixed for GNN)
- **Interactive Chart**: Real-time loss curve visualization

## Troubleshooting

### If the monitor shows "Waiting for simulation data"
1. Check if log files exist: `ls /scratch/{user[0]}/{user}/icon_exercise_comin/log_*.txt`
2. Run `generate_status.py` manually to verify it works
3. Check the simulation is actually running

### If model type shows "Unknown"
- The checkpoint file may not exist yet (only written after first timestep)
- Wait for the first output timestep to complete

### If losses are not updating
- Verify the simulation is still running
- Check that new log files are being created
- Look for error messages in the terminal running the monitor

## Technical Details

### Status JSON Format
```json
{
  "timestamp": "2024-01-28T12:34:56",
  "simulation": {
    "start_time": "2021-07-14 00:00:00",
    "current_time": "2021-07-15 12:00:00",
    "elapsed_time": "1 day, 12:00:00",
    "n_domains": 1,
    "total_points": 20480,
    "output_count": 36
  },
  "training": {
    "model_type": "GNN (Mini-batch)",
    "current_loss": 1.23e-05,
    "total_batches": 144,
    "batches_per_timestep": 4,
    "learning_rate": 0.001,
    "avg_loss": 2.34e-05,
    "min_loss": 1.23e-05,
    "max_loss": 3.45e-05
  }
}
```

## Future Enhancements

Potential improvements for the monitoring system:
- Add node count and edge count for GNN monitoring
- Display graph structure statistics
- Show parameter count for the model
- Add loss convergence rate
- Export training history to CSV
- Add email/Slack notifications for convergence
