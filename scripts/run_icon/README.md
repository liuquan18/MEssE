# ComIn Loss Visualization Tool

## Overview

A Python tool for visualizing ICON-ComIn training loss curves with support for multiple data sources and automatic plotting.

## Quick Start

```bash
cd /work/mh1498/m301257/work/MEssE/scripts/run_icon

# Using job ID (automatically finds corresponding SLURM file)
./quick_plot_loss.sh 22266269

# Using default log directory
./quick_plot_loss.sh

# Custom output filename
./quick_plot_loss.sh -o my_experiment_loss.png
```

## Features

- **Dual-scale plotting**: Linear and logarithmic scales for comprehensive visualization
- **Multiple data sources**: Read from ComIn log files or SLURM output files
- **Automatic statistics**: Displays min, max, mean, final value, and reduction percentage
- **Timestamp support**: Timeline visualization when reading from log directory
- **Flexible output**: Customizable titles, filenames, and data sources

## Installation

### Required Python packages:
```bash
source /work/mh1498/m301257/work/MEssE/environment/python/py_venv/bin/activate
pip install matplotlib numpy
```

## Usage

### Method 1: Quick Script (Recommended)

```bash
# Basic usage with job ID
./quick_plot_loss.sh 22266269

# Specify log directory
./quick_plot_loss.sh -l /scratch/m/m301257/icon_exercise_comin

# Custom output and title
./quick_plot_loss.sh -o experiment1.png -t "Experiment 1 Loss"

# Show help
./quick_plot_loss.sh --help
```

### Method 2: Python Script Directly

```bash
source /work/mh1498/m301257/work/MEssE/environment/python/py_venv/bin/activate

# From ComIn log directory
python plot_comin_loss.py \
    --log-dir /scratch/m/m301257/icon_exercise_comin \
    --output loss_curve.png

# From SLURM output file
python plot_comin_loss.py \
    --slurm-file /path/to/slurm.12345.out \
    --output loss_curve.png

# Custom title
python plot_comin_loss.py \
    --log-dir /scratch/m/m301257/icon_exercise_comin \
    --title "My Experiment - Training Loss" \
    --output custom_loss.png
```

## Command-line Options

### quick_plot_loss.sh options:
```
-h, --help          Show help message
-l, --log-dir DIR   Specify ComIn log directory
-o, --output FILE   Output filename (default: loss_curve.png)
-s, --slurm FILE    Use SLURM output file
-t, --title TITLE   Custom chart title
[job_id]           Job ID (uses slurm.[job_id].out file)
```

### plot_comin_loss.py options:
```
--log-dir PATH      ComIn log file directory
--slurm-file PATH   SLURM output file path
--output, -o FILE   Output image filename
--title TEXT        Chart title
```

## Data Sources

### ComIn Log Files
- **Location**: `/scratch/m/m301257/icon_exercise_comin/log_*.txt`
- **Format**: One floating-point loss value per line
- **Filename**: `log_2021-07-14 00:00:00.txt` (includes timestamp)
- **Advantages**: Complete training history with timestamps

### SLURM Output Files
- **Location**: `/work/mh1498/m301257/work/MEssE/experiment/slurm.*.out`
- **Format**: Lines matching `  0: loss: <value>`
- **Advantages**: Single file, includes other diagnostic info

## Output

The script generates a PNG image with two subplots:

1. **Top panel**: Linear scale loss curve
   - Shows all iterations
   - Statistics box (min, max, mean, final)

2. **Bottom panel**: Logarithmic scale loss curve
   - Better visualization of magnitude changes
   - Shows rapid initial descent clearly

### Example Statistics Output:
```
Loss Statistics:
  Total iterations: 358809
  Minimum: 7.246746e-11
  Maximum: 3.209368e+05
  Mean: 8.955550e-01
  Final: 3.377035e-10
  Std Dev: 5.357812e+02
  Loss reduction: 100.00% (initial vs final)
```

## File Structure

```
scripts/run_icon/
├── plot_comin_loss.py      # Main plotting script
├── quick_plot_loss.sh      # Convenience wrapper script
└── README.md               # This file
```

## Examples

### Example 1: Basic usage
```bash
./quick_plot_loss.sh 22266269
# Output: /work/mh1498/m301257/work/MEssE/experiment/loss_curve.png
```

### Example 2: Compare multiple experiments
```bash
./quick_plot_loss.sh 22266269 -o exp1_loss.png -t "Experiment 1"
./quick_plot_loss.sh 22266500 -o exp2_loss.png -t "Experiment 2"
./quick_plot_loss.sh 22266700 -o exp3_loss.png -t "Experiment 3"
```

### Example 3: Monitor training progress
```bash
# Run periodically during training
watch -n 300 './quick_plot_loss.sh'
```

### Example 4: Use specific log directory
```bash
./quick_plot_loss.sh -l /path/to/different/logs -o custom_loss.png
```

## Viewing Images

### On compute node:
Transfer to local machine:
```bash
scp levante:/work/mh1498/m301257/work/MEssE/experiment/loss_curve.png .
```

### In VS Code:
Navigate to the file in the explorer and click to view.

## Troubleshooting

### Issue: "No loss data to plot"
- Check if log directory exists and contains `log_*.txt` files
- Verify SLURM file path is correct
- Ensure read permissions on files

### Issue: Font warnings
- These are normal and don't affect the plot
- The tool will still generate images successfully

### Issue: Memory error with large datasets
- The tool handles 350K+ data points efficiently
- If issues occur, consider sampling the data

## Integration with Workflow

Suggested workflow:
```bash
# 1. Submit job
cd /work/mh1498/m301257/work/MEssE/scripts/run_icon
bash run_icon_LAM.sh

# 2. Monitor job (in another terminal)
watch -n 60 'squeue -u $USER'

# 3. After job completes, plot loss curve
./quick_plot_loss.sh [job_id]

# 4. Download and view image locally
scp levante:/work/mh1498/m301257/work/MEssE/experiment/loss_curve.png .
```

## Technical Details

- **Language**: Python 3.9+
- **Dependencies**: matplotlib, numpy
- **Image format**: PNG (150 DPI)
- **Default size**: 12" × 10" (two subplots)
- **Color scheme**: Blue (linear), Red (logarithmic)

## License

Part of the MEssE project.

## Support

For issues or questions, please contact the MEssE project team.

---

**Last Updated**: January 27, 2026
