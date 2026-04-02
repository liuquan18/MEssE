# Model Essence Extractor (MEssE)

# Structure
```text
.
├── comin_plugin                # python scripts used as comin plugin
│   ├── MLP_trainer_JAX.py      # simple example of a MLP trainer (ta --> ua) using JAX
│   └── project_Z.py            # simple example of predicting next step (ua_t --> ua_{t+1})
├── monitors                    # *under development*
└── scripts                     # scripts 
    ├── build_env_gpu.sh        # building ICON, ComIn, YAC and preparing python env
    └── run_icon_gpu.sh         # running ICON with A100
```
# Workflow

## Build

Prepare environment using `./scripts/build_env_gpu.sh ${desired_path}`
On successful completion of the script, a py_venv is created under ${desired_path}

## Run
Run ICON with the plugin like so: `./scripts/run_icon_gpu.sh $ICON_BUILD_DIR $COMIN_PLUGIN_PATH $LEVANTE_ACCOUNT`

```bash
`ICON_BUILD_DIR` can be found under `${desired_path}/build_dir`
`COMIN_PLUGIN_PATH` is `$(pwd)/comin_plugin/gnn_trainer.py`
`LEVANTE_ACCOUNT` is your levante project id
```

## Online interface (unavailable for now)

```bash
cd $(pwd)/monitor
./start_monitor.sh 5000 #(any port number)
```

Then access: **http://localhost:5000**
