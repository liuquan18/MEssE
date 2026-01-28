# Model Essence Extractor (MEssE)

# Structure
```bash
.
├── build                       # external location for building ICON
├── data                        # output of icon
├── environment                 # Environment setup scripts
├── experiment                  # Experiment data 
├── ICON                        # ICON source code
└── scripts                      # Scripts 
    ├── build_icon              # scripts for building ICON, ComIn, YAC 
    ├── plugin                  # Python scripts for ComIn     
    └── run_icon                # scripts for running ICON
```
# Workflow

## Build

1. Build python environment `./scripts/build_scripts/build_pyenv.sh`

2. Build ICON `./scripts/build_scripts/build_icon.sh`

If the icon-model is not cloned, the scripts will automatically clone the icon-model from the github repository.

3. Build ComIn `./scripts/build_scripts/build_ComIn.sh`

4. <del> Build YAC `./scripts/build_scripts/build_YAC.sh`



## Run
1. make sure that no bug in plugin `./scripts/plugin/`

2. Prepare the input data `./scripts/run_icon/prepare_data.sh`

3. Run ICON `./scripts/run_icon/run_icon_LAM.sh`, *replace the `--account mh0033` with `--account <your_account>` in the last line of the script.*

the output log file will be in /experiment/ directory.

## Online interface

```bash
cd /work/mh0033/m300883/Project_week/MEssE/scripts/plugin/monitor
./start_monitor.sh
```

Then access: **http://localhost:5000**
