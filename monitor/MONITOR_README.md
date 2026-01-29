# MEssE v1.0 - Monitoring Interface

## Overview
This monitoring interface provides real-time visualization of your ICON simulation and neural network training progress.

## Features
- **Live Simulation Status**: Track ICON simulation time, elapsed time, domains, and grid points
- **Training Metrics**: Monitor training loss in real-time with interactive charts
- **Modern UI**: Clean, responsive interface with live updates
- **Real-time Updates**: Data refreshes every 2 seconds

## Quick Start

### 1. Start the Monitoring Server

```bash
cd /work/mh0033/m300883/Project_week/MEssE/scripts/plugin
./start_monitor.sh
```

The server will start and display access URLs. Look for output like:
```
🚀 Starting monitoring server...

📡 Access URLs:
   Local:    http://localhost:5000
   Network:  http://levante2:5000
```

### 2. Access the Interface

Open your browser and navigate to:
- **Local access**: http://localhost:5000
- **Network access**: http://HOSTNAME:5000 (replace HOSTNAME with your server name)

If you're running this on a remote server (like Levante), you may need to set up SSH port forwarding:

```bash
# On your local machine
ssh -L 5000:localhost:5000 YOUR_USERNAME@levante.dkrz.de
```

Then access: http://localhost:5000 in your local browser

### 3. Run Your ICON Simulation

The monitoring interface will automatically detect and display data from your running ICON simulation with the ComIn plugin.

**Note**: Neural network training begins only after 24 hours of simulation elapsed time. Before that, the interface will show "Training: Waiting" status.

## Interface Layout

```
┌─────────────────────────────────────────────────────┐
│  MEssE v1.0                            ● Live       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────┐  ┌──────────────────────┐   │
│  │ 🌍 ICON          │  │ 🧠 Neural Network    │   │
│  │ Simulation       │  │ Training             │   │
│  │                  │  │                      │   │
│  │ • Start Time     │  │ Current Loss: 0.0123 │   │
│  │ • Current Time   │  │                      │   │
│  │ • Elapsed Time   │  │ • Total Batches      │   │
│  │ • Domains        │  │ • Average Loss       │   │
│  │ • Grid Points    │  │ • Minimum Loss       │   │
│  │ • Output Steps   │  │ • Learning Rate      │   │
│  │                  │  │                      │   │
│  │                  │  │ [Loss Chart]         │   │
│  └──────────────────┘  └──────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Technical Details

### File Structure
```
scripts/plugin/
├── monitor_server.py       # Flask web server
├── start_monitor.sh        # Launcher script
└── templates/
    └── monitor.html        # Web interface
```

### Data Flow
1. The `gnn_trainer_gpu.py` ComIn plugin writes status files to `/scratch/.../icon_exercise_comin/`
2. Training begins only after 24 hours of simulation elapsed time
3. The web server reads these files and serves them via API
4. The browser fetches updates every 2 seconds and displays them

### Requirements
- Python 3.x
- Flask (automatically installed by launcher script)
- Running ICON simulation with ComIn plugin

## Troubleshooting

### Server won't start
- Check if Flask is installed: `python3 -c "import flask"`
- Install manually if needed: `pip install --user flask`

### No data appearing
- Ensure ICON simulation is running
- Check that status files exist: `ls /scratch/.../icon_exercise_comin/monitor_status.json`
- Check server logs for errors

### Port 5000 already in use
Edit `monitor_server.py` and change the port number:
```python
app.run(host='0.0.0.0', port=5001, debug=False)  # Change 5000 to 5001
```

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

## Customization

### Change Update Frequency
Edit `templates/monitor.html`, line with `setInterval`:
```javascript
setInterval(updateStatus, 2000);  // Change 2000 (2 seconds) to desired milliseconds
```

### Modify Chart Appearance
Edit the Chart.js configuration in `templates/monitor.html` starting at line with `new Chart`.

## Support

For issues or questions, check the MEssE project documentation or contact the development team.
