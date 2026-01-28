# ICON ML Training Loss Monitor

A real-time dashboard for monitoring and visualizing loss functions during ICON climate model training with machine learning components.

## Features

🎯 **Real-time Monitoring**
- Live updates of training loss values every 5 seconds
- Automatic data aggregation from log files and TensorBoard events
- No need to restart - picks up new data automatically

📊 **Interactive Visualizations**
- Loss history over time with mean, min, and max values
- Loss distribution histogram
- Per-batch loss analysis for each timestep
- Statistical summaries with percentiles and trends

📈 **Comprehensive Analytics**
- Overall training statistics and trends
- Per-timestep batch analysis
- Loss improvement tracking
- Trend detection (increasing/decreasing)

🌐 **Web-based Interface**
- Modern, responsive dashboard accessible from any browser
- Dark theme optimized for long monitoring sessions
- Works on desktop and mobile devices

## Directory Structure

```
scripts/monitor/
├── dashboard_server.py       # Main Flask server
├── data_aggregator.py        # Data loading and aggregation
├── config.py                 # Configuration settings
├── templates/
│   └── dashboard.html        # Dashboard HTML template
├── static/
│   ├── style.css            # Dashboard styles
│   └── dashboard.js         # Dashboard JavaScript
└── scripts/
    ├── start_monitor.sh     # Start the dashboard
    ├── stop_monitor.sh      # Stop the dashboard
    └── status_monitor.sh    # Check dashboard status
```

## Requirements

### Python Dependencies

**Required:**
- Python 3.7+
- Flask >= 2.0.0
- Flask-CORS >= 3.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0

**Optional:**
- tensorboard >= 2.0.0 (for TensorBoard event file support)

### System Requirements

- Linux system (tested on Levante HPC)
- Access to ICON simulation output directory
- Network connectivity for web dashboard access

## Installation

### 1. Install Python Dependencies

```bash
# Install required packages
pip install --user flask flask-cors pandas numpy

# Install optional TensorBoard support
pip install --user tensorboard
```

### 2. Verify Installation

```bash
cd /work/mh0033/m301250/20260126_ML-ICON/MEssE-1/scripts/monitor
./scripts/status_monitor.sh
```

## Usage

### Starting the Dashboard

Use the convenience script:

```bash
cd /work/mh0033/m301250/20260126_ML-ICON/MEssE-1/scripts/monitor
./scripts/start_monitor.sh
```

Or specify custom host and port:

```bash
./scripts/start_monitor.sh 5001 0.0.0.0
```

Or start directly with Python:

```bash
python3 dashboard_server.py --host 0.0.0.0 --port 5000
```

### Accessing the Dashboard

Once started, open your web browser and navigate to:

```
http://localhost:5000
```

Or from another machine:

```
http://<your-server-ip>:5000
```

**Note:** If running on an HPC system, you may need to:
1. Set up SSH port forwarding: `ssh -L 5000:localhost:5000 user@hpc-node`
2. Or use the HPC's web proxy if available

### Stopping the Dashboard

```bash
./scripts/stop_monitor.sh
```

### Checking Status

```bash
./scripts/status_monitor.sh
```

## Dashboard Components

### 1. Status Bar
- Connection status indicator
- Last update timestamp
- Total timesteps and batches processed

### 2. Statistics Cards
- **Current Loss:** Most recent batch loss value
- **Mean Loss:** Average loss for current timestep
- **Total Timesteps:** Number of simulation timesteps completed
- **Trend:** Whether loss is increasing or decreasing

### 3. Charts

#### Loss History Chart
- Time series of mean, min, and max loss per timestep
- Automatically updates as new data arrives
- Configurable history length (50, 100, 200, 500 points)

#### Loss Distribution Chart
- Histogram showing the distribution of all loss values
- Helps identify training patterns and outliers

#### Batch Losses Chart
- Individual batch losses for the most recent timestep
- Shows loss variation within a single timestep

#### Timestep Statistics Chart
- Mean loss trend with standard deviation bands
- Overview of training progress over time

### 4. Detailed Statistics Table
- Overall statistics (mean, std, min, max)
- Percentiles (25th, 50th, 75th, 90th, 95th, 99th)
- Trend analysis (slope, direction)
- Improvement metrics

### 5. Activity Log
- Real-time log of system events
- Connection status updates
- Error messages and warnings

## Configuration

Edit `config.py` to customize:

```python
# Server settings
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5000

# Update intervals (seconds)
DATA_UPDATE_INTERVAL = 5
DASHBOARD_REFRESH_RATE = 5

# Data settings
DEFAULT_HISTORY_LIMIT = 100
MAX_HISTORY_LIMIT = 1000

# TensorBoard support
TENSORBOARD_ENABLED = True
```

## Data Sources

The monitor automatically reads data from:

1. **Log Files** (`/scratch/<user>/icon_exercise_comin/log_*.txt`)
   - Individual batch loss values
   - Organized by timestamp

2. **Summary Files** (`/scratch/<user>/icon_exercise_comin/summary_*.txt`)
   - Aggregated statistics per timestep
   - Mean, std, min, max values

3. **TensorBoard Events** (optional, `/scratch/<user>/icon_exercise_comin/runs/`)
   - Scalar metrics logged during training
   - Histograms of model parameters

## API Endpoints

The dashboard provides a REST API for programmatic access:

- `GET /api/status` - Overall monitoring status
- `GET /api/loss/current` - Most recent loss values
- `GET /api/loss/history?limit=100` - Historical loss data
- `GET /api/loss/statistics` - Statistical summary
- `GET /api/loss/timesteps` - List of all timesteps
- `GET /api/loss/by_timestep/<timestamp>` - Data for specific timestep
- `GET /api/tensorboard/scalars?tag=Loss/batch` - TensorBoard scalar data
- `GET /api/health` - Health check endpoint

## Troubleshooting

### Dashboard won't start

1. Check if Python and dependencies are installed:
   ```bash
   python3 --version
   python3 -c "import flask, pandas, numpy"
   ```

2. Check if port is already in use:
   ```bash
   lsof -i :5000
   ```

3. Check the log file:
   ```bash
   tail -f /work/mh0033/m301250/20260126_ML-ICON/MEssE-1/scripts/monitor/dashboard.log
   ```

### No data displayed

1. Verify data directory exists:
   ```bash
   ls -la /scratch/${USER:0:1}/$USER/icon_exercise_comin/
   ```

2. Check if ICON simulation is running and producing output

3. Verify log files are being created:
   ```bash
   ls -la /scratch/${USER:0:1}/$USER/icon_exercise_comin/log_*.txt
   ```

### Dashboard shows "Disconnected"

1. Check if the server is running:
   ```bash
   ./scripts/status_monitor.sh
   ```

2. Check network connectivity and firewall settings

3. Try accessing the API directly:
   ```bash
   curl http://localhost:5000/api/health
   ```

### Performance issues

1. Reduce update frequency in `config.py`:
   ```python
   DATA_UPDATE_INTERVAL = 10  # Instead of 5
   ```

2. Limit history display:
   - Use the dropdown to select fewer data points (50 instead of 500)

3. Check system resources:
   ```bash
   top -u $USER
   ```

## Integration with ICON Simulation

The dashboard is designed to work seamlessly with the ICON-ML training pipeline:

1. **ICON runs with ComIn plugin** → Produces loss data
2. **ComIn plugin writes logs** → Creates log and summary files
3. **Dashboard monitors directory** → Automatically picks up new files
4. **Real-time visualization** → Updates every 5 seconds

No manual intervention needed once dashboard is started!

## Advanced Usage

### Running on HPC Login Node

```bash
# Start dashboard on login node
./scripts/start_monitor.sh

# Set up SSH tunnel from local machine
ssh -L 5000:localhost:5000 username@levante.dkrz.de

# Access from local browser at http://localhost:5000
```

### Running as Background Service

The dashboard automatically runs in the background. Check logs:

```bash
tail -f dashboard.log
```

### Custom Data Directory

If your data is in a different location, edit `dashboard_server.py`:

```python
def initialize_aggregator():
    log_dir = Path("/path/to/your/data/directory")
    data_aggregator = LossDataAggregator(log_dir)
```

## Development

### Adding New Metrics

1. Modify `data_aggregator.py` to load new data
2. Add API endpoint in `dashboard_server.py`
3. Update frontend in `static/dashboard.js` to fetch and display

### Customizing Appearance

Edit `static/style.css` to change colors, layout, fonts, etc.

### Adding New Charts

1. Add chart canvas in `templates/dashboard.html`
2. Initialize chart in `static/dashboard.js` (initializeCharts function)
3. Add update logic in updateCharts function

## Performance Considerations

- Dashboard is designed for continuous operation
- Memory usage: ~100-200 MB
- CPU usage: <1% when idle, <5% during updates
- Network: ~10 KB/s data transfer to browser
- Disk I/O: Minimal, only reads log files every 5 seconds

## Security Notes

For production deployment:

1. Set a secret key in `config.py`
2. Use a reverse proxy (nginx, Apache) 
3. Enable HTTPS
4. Restrict access with authentication
5. Use firewall rules to limit access

## License

This monitoring system is part of the MEssE project. See the project's main LICENSE file for details.

## Support

For issues or questions:
- Check the troubleshooting section above
- Review log files in `dashboard.log`
- Check ICON simulation logs for data generation issues

## Acknowledgments

- Built for the ICON climate model training monitoring
- Uses Chart.js for visualization
- Integrates with TensorBoard event files
- Designed for the Levante HPC system at DKRZ

---

**Happy Monitoring! 🌍📊**
