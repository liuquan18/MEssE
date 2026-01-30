# Quick Start Guide: Training and Testing Split

## Summary of Changes

The `diff_trainer.py` script now supports splitting your ICON simulation into:
- **Training phase**: Model trains on incoming data
- **Test phase**: Model freezes, test data saved for evaluation

## Step-by-Step Workflow

### 1. Configure the Split

Edit `/work/mh0033/m301250/20260126_ML-ICON/MEssE-2/comin_plugin/diff_trainer.py` (line ~299):

```python
TRAINING_END_DATE = "1979-01-21T00:00:00"  # Stop training after Jan 20
```

For a 30-day run (Jan 1-31), this gives you:
- **Training**: Days 1-20 (Jan 1-20)
- **Testing**: Days 21-31 (Jan 21-31)

### 2. Run ICON Simulation

```bash
cd /work/mh0033/m301250/20260126_ML-ICON/MEssE-2/scripts
bash launch_icon-diff.sh
```

This will:
- Train the model for 20 days
- Save test data for the last 10 days
- Output locations:
  - Model checkpoint: `/scratch/{user}/icon_exercise_comin/net_1979-01-01T00-00-00.pth`
  - Test data: `/scratch/{user}/icon_exercise_comin/test_data_1979-01-01T00-00-00/`

### 3. Verify Test Data

```bash
cd /work/mh0033/m301250/20260126_ML-ICON/MEssE-2/scripts/validate

python analyze_test_data.py \
    --test-dir /scratch/m/m301250/icon_exercise_comin/test_data_1979-01-01T00-00-00 \
    --export-csv test_summary.csv
```

This will:
- Check data integrity (shapes, NaN values)
- Compute statistics
- Export summary CSV

### 4. Run Model Inference

```bash
python run_inference.py \
    --checkpoint /scratch/m/m301250/icon_exercise_comin/net_1979-01-01T00-00-00.pth \
    --test-dir /scratch/m/m301250/icon_exercise_comin/test_data_1979-01-01T00-00-00 \
    --output-dir /scratch/m/m301250/icon_exercise_comin/predictions \
    --n-samples 10
```

This will:
- Load trained model
- Generate 10 predictions per test sample
- Compute RMSE, MAE, Bias metrics
- Save predictions to output directory

### 5. Analyze Results

Prediction files are saved as:
```
/scratch/{user}/icon_exercise_comin/predictions/
├── predictions_1979-01-21T00-00-00.npz
├── predictions_1979-01-21T00-15-00.npz
├── ...
└── metrics_summary.txt
```

Load and visualize predictions:

```python
import numpy as np

# Load a prediction file
data = np.load('predictions_1979-01-21T00-00-00.npz')

predictions = data['predictions']      # Shape: (80, 10, 256)
ground_truth = data['ground_truth']    # Shape: (80, 256)
coarse_context = data['coarse_context']  # Shape: (80, 7)

# Example: Compare prediction mean with ground truth
pred_mean = predictions.mean(axis=1)   # Average over 10 samples
rmse = np.sqrt(((pred_mean - ground_truth) ** 2).mean())
print(f"RMSE: {rmse:.6e}")
```

## Key Configuration Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `TRAINING_END_DATE` | `diff_trainer.py` line ~299 | `"1979-01-21T00:00:00"` | Date to stop training |
| `START_DATE` | `launch_icon-diff.sh` line 9 | `"1979-01-01T00:00:00Z"` | Simulation start |
| `END_DATE` | `launch_icon-diff.sh` line 10 | `"1979-01-31T00:00:00Z"` | Simulation end |
| `BUFFER_SIZE` | `diff_trainer.py` line 403 | 48 | Training buffer (timesteps) |

## Timeline Example (30-day run)

```
Jan 1 00:00 ─────────────── Jan 21 00:00 ─────────────── Jan 31 00:00
     │                           │                           │
     └─── Training Phase ────────┘                           │
             (20 days)                                       │
                                 └──── Test Phase ───────────┘
                                         (10 days)
```

## Troubleshooting

### "No test data files found"
- Ensure ICON ran past `TRAINING_END_DATE`
- Check SLURM output for errors
- Verify `END_DATE > TRAINING_END_DATE` in config

### "Model checkpoint not found"
- Training needs at least 12 hours (48 timesteps) to create first checkpoint
- Check: `/scratch/{user}/icon_exercise_comin/net_*.pth` exists
- Review training logs: `/scratch/{user}/icon_exercise_comin/log_session_*.txt`

### "Import error: No module named 'diff_trainer'"
- Make sure you're in the correct directory when running scripts
- The inference script adds the comin_plugin directory to Python path automatically

## What Gets Saved

### Training Phase Output
- `net_{start_time}.pth` - Model checkpoint (contains weights, optimizer state, EMA stats)
- `log_session_{counter}_{time}.txt` - Training loss logs

### Test Phase Output  
- `test_data_{start_time}/test_sample_{timestamp}.npz` - One file per ICON output step
  - Contains: normalized/unnormalized coarse+fine data, EMA parameters, timestamp

### Inference Output
- `predictions_{timestamp}.npz` - Model predictions + ground truth
- `metrics_summary.txt` - Evaluation metrics (RMSE, MAE, Bias)

## Need More Details?

See the full documentation: `/work/mh0033/m301250/20260126_ML-ICON/MEssE-2/comin_plugin/TRAINING_TEST_SPLIT.md`
