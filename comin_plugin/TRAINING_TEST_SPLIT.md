# Training and Test Data Split Configuration

## Overview

The `diff_trainer.py` script now supports splitting the simulation into **training** and **testing** phases. This allows you to:
- Train the diffusion model on the first N days of data
- Save the remaining days as test data for model evaluation

## Configuration

### Setting the Training Cutoff Date

Edit the `TRAINING_END_DATE` variable in `diff_trainer.py` (line ~299):

```python
TRAINING_END_DATE = "1979-01-21T00:00:00"  # Stop training after Jan 20, 1979 23:59:59
```

### Example: 30-Day Simulation (20 days training, 10 days testing)

In `scripts/launch_icon-diff.sh`:
```bash
START_DATE="1979-01-01T00:00:00Z"  # Start date
END_DATE="1979-01-31T00:00:00Z"    # End date (31 days total)
```

In `comin_plugin/diff_trainer.py`:
```python
TRAINING_END_DATE = "1979-01-21T00:00:00"  # Train only first 20 days
```

This configuration will:
- **Days 1-20 (Jan 1-20)**: Model trains normally, updates weights, saves checkpoints
- **Days 21-31 (Jan 21-31)**: Model freezes, test data saved to disk

## How It Works

### Training Phase (before cutoff)
- Data is normalized and added to replay buffer
- Model trains every 15 minutes (ICON output interval)
- Checkpoints saved: `/scratch/{user}/icon_exercise_comin/net_{start_time}.pth`
- Training logs: `/scratch/{user}/icon_exercise_comin/log_session_{counter}_{time}.txt`

### Test Phase (after cutoff)
- **No training occurs** (model weights frozen)
- Each timestep's data is saved to disk:
  ```
  /scratch/{user}/icon_exercise_comin/test_data_{start_time}/test_sample_{timestamp}.npz
  ```
- Saved data includes:
  - `coarse_context_norm`: Normalized coarse context (80, 7) - model input
  - `fine_field_norm`: Normalized fine field (80, 256) - ground truth
  - `coarse_context`: Unnormalized coarse data (for visualization)
  - `fine_field`: Unnormalized fine data (for visualization)
  - Normalization parameters (EMA mean/std)
  - Timestamp metadata

## Using Test Data

### Loading Test Data in Python

```python
import numpy as np
import glob

# Load all test samples
test_dir = "/scratch/{user}/icon_exercise_comin/test_data_1979-01-01T00-00-00"
test_files = sorted(glob.glob(f"{test_dir}/test_sample_*.npz"))

for test_file in test_files:
    data = np.load(test_file)
    
    # Get normalized inputs (for model inference)
    coarse_context_norm = data['coarse_context_norm']  # Shape: (80, 7)
    fine_field_norm = data['fine_field_norm']          # Shape: (80, 256)
    
    # Get unnormalized data (for visualization/evaluation)
    coarse_context = data['coarse_context']            # Shape: (80, 7)
    fine_field = data['fine_field']                    # Shape: (80, 256)
    
    # Get normalization parameters
    ema_coarse_mean = data['ema_coarse_mean']
    ema_coarse_std = data['ema_coarse_std']
    ema_fine_mean = data['ema_fine_mean']
    ema_fine_std = data['ema_fine_std']
    
    print(f"Timestamp: {data['simulation_time']}")
```

### Running Model Inference on Test Data

```python
import torch
from diff_trainer import ConditionalDiffusionDownscaler

# Load trained model
checkpoint = torch.load("net_1979-01-01T00-00-00.pth", weights_only=False)
model = ConditionalDiffusionDownscaler(n_fine_cells=256, n_neighbors=7, n_hidden=256, n_timesteps=50)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test sample
data = np.load("test_sample_1979-01-21T00-00-00.npz")
coarse_context_norm = torch.FloatTensor(data['coarse_context_norm'])  # (80, 7)

# Generate predictions (multiple samples for uncertainty quantification)
with torch.no_grad():
    predictions = model.sample(coarse_context_norm, n_samples=10)  # Shape: (80, 10, 256)

# Denormalize predictions
ema_fine_mean = data['ema_fine_mean']
ema_fine_std = data['ema_fine_std']
predictions_denorm = predictions * ema_fine_std + ema_fine_mean

# Compare with ground truth
ground_truth = data['fine_field']  # Shape: (80, 256)
```

## Output Directory Structure

```
/scratch/{user}/icon_exercise_comin/
├── net_1979-01-01T00-00-00.pth              # Model checkpoint (training phase)
├── log_session_0001_*.txt                   # Training logs
├── log_session_0002_*.txt
├── ...
└── test_data_1979-01-01T00-00-00/           # Test data directory
    ├── test_sample_1979-01-21T00-00-00.npz  # First test sample
    ├── test_sample_1979-01-21T00-15-00.npz  # Second test sample (15 min later)
    ├── test_sample_1979-01-21T00-30-00.npz
    └── ...                                   # One file per ICON output timestep
```

## Important Notes

1. **EMA Statistics**: The normalization parameters saved in test data are based on EMA (Exponential Moving Average) statistics accumulated during training. These should be used to denormalize model predictions.

2. **Grid Structure**: 
   - Coarse context: (80, 7) = 80 R2B01 cells × [center + 6 neighbors]
   - Fine field: (80, 256) = 80 coarse cells × 256 fine cells per coarse cell

3. **Memory Usage**: Test data files are ~170 KB each. For 10 days at 15-minute intervals: 10 days × 96 timesteps/day × 170 KB ≈ 163 MB total.

4. **Modify Cutoff Date**: Change `TRAINING_END_DATE` in `diff_trainer.py` to adjust train/test split.

## Troubleshooting

### Issue: Test data directory is empty
- Check that simulation ran past the cutoff date
- Verify `END_DATE` in `launch_icon-diff.sh` is after `TRAINING_END_DATE`
- Check SLURM logs for errors

### Issue: Model checkpoint not found during test phase
- Ensure training phase completed successfully
- Check that buffer filled completely (requires 48 timesteps = 12 hours)
- Verify checkpoint path: `/scratch/{user}/icon_exercise_comin/net_{start_time}.pth`

### Issue: NaN values in test data
- Check ICON output for issues
- Verify near-surface air temperature variable is correctly specified in `diff_trainer.py`
- Review EMA statistics in training logs
