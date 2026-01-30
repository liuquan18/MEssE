# Training Strategy Update

## Changes Made

### New Training Strategy
The diffusion model training has been updated to use a **continuous learning approach** instead of periodic training:

**Previous Strategy:**
- Train every 12 callbacks (every 3 hours)
- Risk missing training sessions if simulation restarts

**New Strategy:**
- **Wait until buffer is full** (48 timesteps = 12 hours of data)
- **Then train on EVERY callback** with the rolling window of latest 48 timesteps
- More frequent updates = better adaptation to changing patterns

### Benefits
1. ✅ **No missed training sessions** - trains continuously once buffer is full
2. ✅ **Better adaptation** - model updates with every new data point
3. ✅ **Consistent tracking** - training session counter always increments sequentially
4. ✅ **Clearer logs** - `log_session_XXXX_timestamp.txt` format

### Files Modified

#### 1. `comin_plugin/diff_trainer.py`
- Removed `TRAIN_INTERVAL` constant
- Added buffer fullness check: `if len(replay_buffer) < BUFFER_SIZE: skip training`
- Training counter now increments on every training (after buffer is full)
- Log files now include training session number: `log_session_0001_1979-01-03T12-00-00.txt`
- Log format includes:
  ```
  training_session: 1
  simulation_time: 1979-01-03T12-00-00
  epoch_1: 0.123456
  epoch_2: 0.112345
  epoch_3: 0.101234
  ```

#### 2. `scripts/validate/plot_loss.py`
- Updated to support both old and new log formats
- Now creates **two plots**:
  - Loss vs Time (hours) - shows temporal evolution
  - Loss vs Training Session - shows sequential progression
- Automatically detects log format (old vs new)

## Expected Behavior

### First 48 Callbacks (Buffer Filling Phase)
```
Callback 1: Buffer filling: 1/48 timesteps collected
Callback 2: Buffer filling: 2/48 timesteps collected
...
Callback 47: Buffer filling: 47/48 timesteps collected
```

### After Buffer is Full (Continuous Training)
```
Callback 48: Training session 1 at 1979-01-03T12:00:00: 48 timesteps = 3840 samples
Callback 49: Training session 2 at 1979-01-03T12:15:00: 48 timesteps = 3840 samples
Callback 50: Training session 3 at 1979-01-03T12:30:00: 48 timesteps = 3840 samples
...
```

### Training Timeline
- **Buffer size**: 48 timesteps (12 hours @ 15min intervals)
- **Training frequency**: Every callback after buffer is full
- **Training sessions**: Sequential, no gaps
- **Checkpoint**: Updated after every callback (with or without training)

## How to Visualize

Run the updated plotting script:
```bash
cd /work/mh1498/m301257/project2/MEssE/scripts/validate
python plot_loss.py
```

Output: `/scratch/m/m301257/icon_exercise_comin/loss_evolution.png`

## Migration Notes

- Old log files (`log_*.txt`) will still be readable
- New runs will create `log_session_XXXX_*.txt` files
- Both formats contain epoch losses, so plotting works with both
- Training counter persists in checkpoint, so it continues correctly after restarts
