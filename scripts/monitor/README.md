# ICON Training Monitoring Tools

This directory contains tools for monitoring and visualizing the training process of the neural network embedded in the ICON climate model via the ComIn plugin.

## Overview

The monitoring system provides **real-time** and **post-hoc** analysis of the training process, including:
- Loss curves and convergence analysis
- Training statistics and trends
- TensorBoard integration for deep model inspection
- Interactive dashboards for live monitoring

## Generated Data

During ICON model execution with the plugin, the following data is generated in:
```
/scratch/<first_letter_of_username>/<username>/icon_exercise_comin/
```

### Files:
- `log_<timestamp>.txt` - Raw loss values for each batch at each timestep
- `summary_<timestamp>.txt` - Statistics summary for each timestep (mean, std, min, max loss)
- `net_<start_time>.pth` - Model checkpoint (weights + optimizer state)
- `runs/experiment_<start_time>/` - TensorBoard logs directory

## Tools

### 1. TensorBoard (Real-time)
**Best for:** Professional-grade real-time monitoring with rich visualizations

Launch TensorBoard server to visualize training metrics as they are generated:

```bash
./launch_tensorboard.sh [PORT]
```

**Default port:** 6006

**Access:**
- Local: `http://localhost:6006`
- Remote: Create SSH tunnel first:
  ```bash
  ssh -L 6006:localhost:6006 <username>@<server>
  ```
  Then open: `http://localhost:6006`

**Features:**
- Loss curves (per batch and per timestep)
- Learning rate tracking
- Model parameter histograms
- Gradient flow visualization
- Scalable and professional interface

---

### 2. Static Plot Generator (Post-hoc)
**Best for:** Publication-quality figures and detailed analysis after training

Generate comprehensive static plots from all log files:

```bash
./plot_training_loss.py
```

**Output location:** `/scratch/<first_letter>/<username>/icon_exercise_comin/plots/`

**Generated plots:**
1. `loss_vs_batch.png` - Loss trajectory over all batches
2. `loss_vs_time.png` - Loss trajectory over simulation time
3. `loss_distribution.png` - Histogram of loss values
4. `loss_smoothed.png` - Loss with moving average overlay
5. `mean_loss_per_timestep.png` - Mean loss per timestep with error bars
6. `loss_range_per_timestep.png` - Min/max loss range per timestep
7. `batches_per_timestep.png` - Number of batches processed per timestep

**Also prints:** Comprehensive statistics including convergence analysis

**Requirements:**
- matplotlib
- pandas
- numpy

---

### 3. Interactive Dashboard (Real-time)
**Best for:** Quick real-time monitoring during training with auto-refresh

Launch an interactive matplotlib-based dashboard that auto-refreshes:

```bash
./training_dashboard.py
```

**Features:**
- Auto-refresh every 5 seconds
- 6-panel dashboard with:
  - Loss vs batch number
  - Loss vs simulation time
  - Loss distribution histogram
  - Mean loss per timestep (with error bars)
  - Training statistics panel
  - Recent loss trend (last 100 batches)
- Live statistics display
- Trend analysis

**Note:** Requires X11 forwarding or VNC for remote visualization
```bash
ssh -X <username>@<server>  # For X11 forwarding
```

---

## Workflow

### During Training (Real-time monitoring)

**Option A: TensorBoard (Recommended)**
```bash
# Terminal 1: Start ICON model with plugin
cd /work/mh0033/m301250/20260126_ML-ICON/MEssE-1/scripts/run_icon
./run_icon_LAM.sh

# Terminal 2: Launch TensorBoard
cd /work/mh0033/m301250/20260126_ML-ICON/MEssE-1/scripts/monitor
./launch_tensorboard.sh
```

**Option B: Interactive Dashboard**
```bash
# Terminal 1: Start ICON model with plugin
cd /work/mh0033/m301250/20260126_ML-ICON/MEssE-1/scripts/run_icon
./run_icon_LAM.sh

# Terminal 2: Launch dashboard (requires X11)
cd /work/mh0033/m301250/20260126_ML-ICON/MEssE-1/scripts/monitor
./training_dashboard.py
```

### After Training (Post-hoc analysis)

```bash
cd /work/mh0033/m301250/20260126_ML-ICON/MEssE-1/scripts/monitor
./plot_training_loss.py
```

This generates all publication-ready plots in the `plots/` directory.

---

## Understanding the Metrics

### Loss (MSE)
- **Metric:** Mean Squared Error between predicted and actual cloud ice content
- **Lower is better:** Model predictions are closer to ground truth
- **Expected behavior:** Should decrease over time as model learns

### Convergence Indicators
- **Decreasing trend:** Model is learning effectively
- **Plateau:** Model has reached optimal performance or needs adjustment
- **Increasing trend:** May indicate learning rate issues or overfitting

### Per-timestep Statistics
- **Mean loss:** Average performance across all batches in a timestep
- **Std dev:** Variability in performance (lower = more stable)
- **Min/max range:** Performance bounds

---

## Troubleshooting

### No data found
**Problem:** Tools report no log files found

**Solution:**
- Ensure ICON model has been run with the plugin
- Check that the scratch directory exists
- Verify username in path

### TensorBoard not starting
**Problem:** `tensorboard: command not found`

**Solution:**
```bash
# Load Python environment with TensorBoard installed
module load python3
pip install tensorboard --user
```

### Dashboard window not appearing
**Problem:** No GUI window when running `training_dashboard.py`

**Solution:**
- Enable X11 forwarding: `ssh -X <username>@<server>`
- Or use VNC session
- Or use TensorBoard instead (no X11 required)

### Slow performance
**Problem:** Dashboard or plots are slow with large datasets

**Solution:**
- The dashboard automatically limits to 1000 points for performance
- For very large runs, use TensorBoard (optimized for large-scale data)

---

## Dependencies

All tools require:
- Python 3.7+
- numpy
- pandas
- matplotlib

TensorBoard additionally requires:
- tensorboard (automatically installed with PyTorch)

These should already be available in the ICON Python environment created by `build_pyenv.sh`.

---

## Customization

### Change TensorBoard port
```bash
./launch_tensorboard.sh 8080  # Use port 8080 instead
```

### Adjust dashboard refresh rate
Edit `training_dashboard.py` and change:
```python
REFRESH_INTERVAL = 5  # seconds (line 16)
```

### Modify plot styles
Edit `plot_training_loss.py` to customize:
- Figure sizes
- Colors
- Plot types
- Statistics displayed

---

## Integration with Plugin

The monitoring tools work seamlessly with the modified `comin_plugin.py`, which now includes:

1. **TensorBoard SummaryWriter** - Logs metrics during training
2. **Enhanced logging** - Saves detailed statistics per timestep
3. **Checkpoint tracking** - Saves global step count for resumption

All monitoring is performed on **rank 0 only** (MPI master process) to avoid data duplication.

---

## Quick Reference

| Tool | Use Case | Output | Real-time |
|------|----------|--------|-----------|
| `launch_tensorboard.sh` | Professional monitoring | Web interface | ✅ Yes |
| `plot_training_loss.py` | Publication plots | PNG files | ❌ No |
| `training_dashboard.py` | Quick visual check | Interactive GUI | ✅ Yes |

---

## Example Output

After running `plot_training_loss.py`, you'll see:
```
==================================================
ICON Training Loss Visualization
==================================================
Log directory: /scratch/m/m301250/icon_exercise_comin
Output directory: /scratch/m/m301250/icon_exercise_comin/plots

Parsing log files...
Found 48 log files
Parsing summary files...
Found 48 summary files

============================================================
TRAINING STATISTICS
============================================================

Total batches processed: 9600
Loss statistics:
  Mean:   0.00123456
  Median: 0.00120345
  Std:    0.00056789
  Min:    0.00045678
  Max:    0.00456789

Timesteps processed: 48
  First: 2021-07-14 00:00:00
  Last:  2021-07-15 23:00:00

Convergence analysis:
  First half mean loss:  0.00145678
  Second half mean loss: 0.00101234
  Improvement:           30.50%
============================================================

Generating plots...
Saved: /scratch/m/m301250/icon_exercise_comin/plots/loss_vs_batch.png
Saved: /scratch/m/m301250/icon_exercise_comin/plots/loss_vs_time.png
...

All plots saved to: /scratch/m/m301250/icon_exercise_comin/plots
Done!
```

---

## Support

For issues or questions:
1. Check TensorBoard logs for detailed training information
2. Verify log files are being created in the scratch directory
3. Ensure Python environment has all required packages
4. Review ICON model stderr output for plugin messages

---

**Last updated:** January 28, 2026
