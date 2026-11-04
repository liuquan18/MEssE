# MEssE v1.0 - Quick Start Guide

## ✅ Server is Running!

Your monitoring server is currently active at:

**🌐 Access URLs:**
- **Local**: http://localhost:5000
- **Network**: http://136.172.124.3:5000
- **Or**: http://levante2:5000

## 📋 What's Working

✅ Flask installed and configured
✅ Web server running on port 5000
✅ Monitoring interface ready
✅ Python environment activated

## 🚀 How to Use

### Option 1: Access from Levante (Same Machine)
Simply open a browser on the Levante system and navigate to:
```
http://localhost:5000
```

### Option 2: Access from Your Local Computer
If you're connecting to Levante remotely, set up SSH port forwarding:

**On your local machine, run:**
```bash
ssh -L 5000:localhost:5000 m300883@levante.dkrz.de
```

Then open your browser and go to:
```
http://localhost:5000
```

### Option 3: Direct Network Access
If you're on the same network, use:
```
http://136.172.124.3:5000
```

## 📊 Interface Features

Once opened, you'll see:

```
┌─────────────────────────────────────────┐
│  MEssE v1.0              ● Live         │
├─────────────────────────────────────────┤
│  🌍 ICON Simulation  │  🧠 NN Training  │
│  • Start Time        │  • Current Loss  │
│  • Current Time      │  • Loss Chart    │
│  • Elapsed Time      │  • Total Batches │
│  • Domains           │  • Avg/Min Loss  │
│  • Grid Points       │                  │
└─────────────────────────────────────────┘
```

- **Left Column**: Real-time ICON simulation information
- **Right Column**: Neural network training loss with live chart
- **Auto-refresh**: Updates every 2 seconds

## 🔄 Managing the Server

### To Stop the Server
Press `Ctrl+C` in the terminal where it's running

### To Restart the Server
```bash
cd /work/mh0033/m300883/Project_week/MEssE/scripts/plugin
source /work/mh0033/m300883/Project_week/MEssE/build/messe_env/py_env/bin/activate
python monitor_server.py
```

Or use the launcher script:
```bash
cd /work/mh0033/m300883/Project_week/MEssE/scripts/plugin
./start_monitor.sh
```

## 📁 Files Created

- `monitor_server.py` - Flask web server
- `templates/monitor.html` - Web interface
- `start_monitor.sh` - Launcher script
- `comin_plugin.py` - Updated to write monitoring data

## 📝 Data Location

The server reads monitoring data from:
```
/scratch/m/m300883/icon_exercise_comin/
├── monitor_status.json     # Current simulation status
└── log_*.txt              # Training loss history
```

## 🔍 Troubleshooting

### "Waiting for simulation data..."
This is normal when:
- ICON simulation hasn't started yet
- First output timestep hasn't occurred yet
- Status files haven't been created

**Solution**: Start your ICON simulation with the ComIn plugin, and data will appear after the first output timestep.

### Can't access the URL
- Verify the server is still running (check terminal)
- For remote access, ensure SSH port forwarding is set up
- Check firewall settings if using direct network access

### Port 5000 already in use
Stop other services using port 5000 or edit `monitor_server.py` to use a different port.

## 🎯 Next Steps

1. ✅ Server is running
2. 🔄 Access the interface in your browser
3. 🚀 Run your ICON simulation
4. 📊 Watch real-time monitoring

Enjoy monitoring your simulation! 🎉
