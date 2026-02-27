# Model Essence Extractor (MEssE)

# Structure
```bash
.
├── comin_plugin                # python scripts used as comin plugin
    ├── gnn_model.py            # where the GNN is defined.
    ├── gnn_trainer.py          # scripts to train the GNN.
    ├── minimal_trainer.py      # a very simple FCN to train.
    ├── utils.py                # some help function.
├── monitors                    # monitoring scripts for online interface
└── scripts                     # scripts 
    ├── build_env.sh            # script for building ICON, ComIn, YAC and preparing python env
    ├── run_icon_gpu.sh         # script for running ICON with A100
    └── run_icon                # script for running ICON
```
# Workflow

## Build

Prepare environment using `./scripts/build_env.sh ${desired_path}`
On successful completion of the script, a py_venv is created under ${desired_path}

## Run
Run ICON with the plugin like so: `./scripts/run_icon_gpu.sh $ICON_BUILD_DIR $COMIN_PLUGIN_PATH $LEVANTE_ACCOUNT`

`ICON_BUILD_DIR` can be found under `${desired_path}/build_dir`

`COMIN_PLUGIN_PATH` is `$(pwd)/comin_plugin/gnn_trainer.py`

`LEVANTE_ACCOUNT` is your levante project id

## Online interface

```bash
cd $(pwd)/monitor
./start_monitor.sh 5000 #(any port number)
```

Then access: **http://localhost:5000**
