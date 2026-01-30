# Loss Function Analysis Scripts

This directory contains tools for extracting, saving, and visualizing training loss from the diffusion model.

## Main Script: `extract_loss.py`

A comprehensive tool that combines loss extraction, data saving, and visualization.

### Features

- ✅ **Extract** loss values from all training session logs
- 💾 **Save** to JSON and CSV formats for analysis
- 📊 **Visualize** with dual-panel plots (loss vs time & loss vs session)
- 📝 **Summary** statistics printed to console

### Usage Examples

```bash
# Basic: Extract and print summary
python extract_loss.py

# Extract, save files, and generate plot
python extract_loss.py --plot

# Show plot interactively (in addition to saving)
python extract_loss.py --plot --show

# Custom output directory
python extract_loss.py --plot --output-dir ./results

# Custom log directory
python extract_loss.py --log-dir /path/to/logs --plot

# Only plot, don't save JSON/CSV
python extract_loss.py --plot --no-save
```

### Output Files

- `loss_history.json` - Structured loss data with all epochs
- `loss_history.csv` - Spreadsheet-friendly format
- `loss_evolution.png` - Dual-panel visualization

### Command-Line Options

```
--log-dir PATH        Directory containing log files (default: /scratch/<user>/icon_exercise_comin/)
--output-dir PATH     Directory for output files (default: current directory)
--output-json FILE    Output JSON file name (default: loss_history.json)
--output-csv FILE     Output CSV file name (default: loss_history.csv)
--plot                Generate loss evolution plot
--plot-file FILE      Plot output file name (default: loss_evolution.png)
--show                Display plot interactively (requires --plot)
--no-save             Skip saving JSON/CSV files, only generate plot
```

## Backward Compatibility

- `plot_loss.py` is a symlink to `extract_loss.py`
- Old script backed up as `plot_loss_old.py.bak`

## Data Format

### JSON Format
```json
[
  {
    "session": 1,
    "timestamp": "1979-01-03T12-00-00",
    "epoch_losses": [0.123456, 0.098765, 0.087654],
    "final_loss": 0.087654,
    "log_file": "/path/to/log_session_0001_1979-01-03T12-00-00.txt"
  },
  ...
]
```

### CSV Format
```
session,timestamp,epoch,loss
1,1979-01-03T12-00-00,1,0.123456
1,1979-01-03T12-00-00,2,0.098765
1,1979-01-03T12-00-00,3,0.087654
...
```

## Visualization

The plot shows two panels:
1. **Left panel**: Loss vs Training Time (hours) - shows temporal evolution
2. **Right panel**: Loss vs Training Session - shows discrete training events

Both panels include:
- Final epoch loss (solid line with circles)
- First epoch loss if multiple epochs (dashed line with squares)
- Statistics box with latest, minimum loss, and session count
