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
    └── run_icon_gpu.sh                 # patches an ICON runscript and submits it
```
Each `plugin.py` is run by ComIn by its path, given in the runscript as `plugin_list(1)%options`.
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
First copy the prepared template `exp.atm_nwp_jsbach_xpp_r2b4` to the `run` folder under the build directory of ICON, then create the run script using `./make_runscripts --all`.

Run ICON with the plugin like so: `./scripts/run_icon_gpu.sh $ICON_BUILD_DIR $COMIN_PLUGIN_PATH $LEVANTE_ACCOUNT`


> `ICON_BUILD_DIR` is `${desired_path}/messe_env/build_dir/icon-model`  
> `COMIN_PLUGIN_PATH` is e.g. `$(pwd)/messe/reconstructor_forecaster/plugin.py`  
> `LEVANTE_ACCOUNT` is your levante project id

LOG file can be found in `${ICON_BUILD_DIR}/run/exp.atm_nwp_jsbach_xpp_r2b4*.o`


## Online interface 

```bash
./scripts/monitor.sh $LOG_FILE_PATH $local_host_port
```

Then access: **http://localhost:$local_host_port**
