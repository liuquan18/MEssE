# Model Essence Extractor (MEssE)
A framework for pretraining surrogate AI models online for km-scale ICON simulations. Documentation and Demo are available in the [overview](./docs/overview.md) page.

# Structure
```text
.
├── pyproject.toml                      # installs the package `messe` (pip install -e .)
├── messe/
│   ├── common/                         # used by more than one model
│   │   ├── icon_data.py                # ComIn arrays <-> tensors, MPI/DDP setup, ICON's clock
│   │   ├── training.py                 # normalization from the dry run, checkpoints, log averaging
│   │   └── nested_grid.py              # rank patches of the nested ICON grids, cell exchange
│   ├── reconstructor_forecaster/       # the main model (RF)
│   │   ├── plugin.py                   # the ComIn plugin
│   │   ├── model.py                    # reconstructor, forecaster, online trainer
│   │   ├── config.py                   # reads config/*.yaml
│   │   ├── kv_memory.py                # key/value memory of the forecaster
│   │   └── forcings.py                 # TOA insolation, cell position
│   ├── other_models/
│   │   ├── autoencoder/                # one-stage FieldSpace autoencoder (plugin.py, model.py)
│   │   ├── gnn/                        # GNN on the ICON native grid (plugin.py, model.py, graph.py)
│   │   ├── unet/                       # UNet on HEALPix via YAC (plugin.py, model.py, healpix.py)
│   │   └── jax/                        # JAX version, under development
│   └── diagnostics/                    # decomposition probe plugin, RF GPU benchmark
├── config/                             # RF configurations (YAML)
├── tests/                              # pytest, CPU only
├── monitor/                            # web monitor of a run's log
└── scripts/
    ├── build_env_gpu.sh                # builds ICON, ComIn, YAC and the python env
    ├── amip-aes_R2B8_gpu.config        # mkexp experiment: AES R2B8 on GPUs with the plugin
    └── run_icon_gpu.sh                 # creates the experiment with mkexp and submits it
```
Each `plugin.py` is run by ComIn by its path, given in the mkexp config as `comin_plugin_nml` `options` (ComIn 1.0).
It imports everything else as `messe.*`, so MEssE must be installed in the Python env.

# Workflow

## Build

Prepare environment using `./scripts/build_env_gpu.sh ${desired_path}`

To use the FieldSpace NN, We suggest to go to the root folder which include `MEssE`, git clone the project (more info [here](https://github.com/FREVA-CLINT/FieldSpaceNN))
```bash
git clone https://github.com/FREVA-CLINT/FieldSpaceNN.git
source ${desired_path}/messe_env/py_env/bin/activate  # activate the python environment
pip install --no-deps -e FieldSpaceNN
pip install --no-deps --no-build-isolation -e MEssE  # the package `messe` used by the plugins
```
`build_env_gpu.sh` runs both installs if FieldSpaceNN sits next to MEssE.

## Run
The experiment is made with ICON's mkexp from `scripts/amip-aes_R2B8_gpu.config`
(standard experiment `amip-aes-R2B8` plus the plugin, its python env and the output stream):

```bash
./scripts/run_icon_gpu.sh $ICON_DIR scripts/amip-aes_R2B8_gpu.config $COMIN_PLUGIN_PATH $LEVANTE_ACCOUNT $EXPERIMENT_NAME [NUM_NODES] [KEY=VALUE ...]
```

> `ICON_DIR` is the directory given to `build_env_gpu.sh` (with `src_code/`, `build_dir/`, `py_env/`)  
> `COMIN_PLUGIN_PATH` is e.g. `$(pwd)/messe/reconstructor_forecaster/plugin.py`  
> `LEVANTE_ACCOUNT` is your levante project id  
> `KEY=VALUE` are further mkexp settings, e.g. `jobs.run.time_limit=00:30:00` or `MESSE_RF_CONFIG_FILE=$(pwd)/config/<file>.yaml`  
> `NO_SUBMIT=1` creates the experiment scripts without submitting them

The python adapter of ComIn must be linked against the Python of `py_env` (configure argument
`PYTHON=$ICON_DIR/py_env/bin/python3`, as in `build_env_gpu.sh`); the script checks this.

The job log is `<LOG_DIR>/<EXPERIMENT_NAME>.run.<jobid>.log`, linked as
`<SCRIPT_DIR>/<EXPERIMENT_NAME>.run.log`; ICON runs (and writes its output and the plugin's
`saved_models/`) in `<WORK_DIR>/run_<start>-<end>/`. Print the directories with
`getexp -k SCRIPT_DIR -k WORK_DIR -k LOG_DIR scripts/amip-aes_R2B8_gpu.config EXP_ID=$EXPERIMENT_NAME`
after `module use $ICON_DIR/src_code/icon-model/etc/Modules; module load icon-levante`.


## Online interface 

```bash
./scripts/monitor.sh $LOG_FILE_PATH $local_host_port
```

Then access: **http://localhost:$local_host_port**
