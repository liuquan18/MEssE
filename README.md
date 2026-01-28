# Model Essence Extractor (MEssE)

# Structure
```bash
.
├── comin_plugin                # python scripts used as comin plugin
└── scripts                     # scripts 
    ├── build_env.sh            # script for building ICON, ComIn, YAC and preparing python env
    └── run_icon                # script for running ICON
```
# Workflow

## Build

Prepare environment using `./scripts/build_env.sh ${desired_path}`
On successful completion of the script, a py_venv is created under ${desired_path}

## Run
Run ICON with the plugin like so: `./scripts/run_icon.sh $ICON_BUILD_DIR $COMIN_PLUGIN_PATH $LEVANTE_ACCOUNT`

`ICON_BUILD_DIR` can be found under `${desired_path}/build_dir`
`COMIN_PLUGIN_PATH` is `$(pwd)/comin_plugin/minimal_trainer.py`
`LEVANTE_ACCOUNT` is your levante project id