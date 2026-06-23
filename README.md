# Model Essence Extractor (MEssE)
A framework for pretraining surrogate AI models online for km-scale ICON simulations. Documentation and Demo are available in the [overview](./docs/overview.md) page.

# Structure
```text
.
├── comin_plugin_JAX                    # project in JAX *under development*
├── comin_plugin_torch                  # in Pytorch 
│   ├── fieldspacenn_online.py          # wrapper of [FieldSpaceNN](https://github.com/FREVA-CLINT/FieldSpaceNN)
│   ├── fieldspacenn_plugin.py          # the plugin to be connected to COMIN
│   ├── unet_online.py                  # the plugin to be connected to COMIN
│   └── unet_plugin.py                  # data preparation and online training
├── monitors                            # *under development*
└── scripts                             # scripts 
    ├── build_env_gpu.sh                # building ICON, ComIn, YAC and preparing python env
    ├── run_icon_gpu.sh                 # running ICON with GPU
    └── exp.aes_amip_messe_test.run     # prepared running scripts 
```
# Workflow

## Build

Prepare environment using `./scripts/build_env_gpu.sh ${desired_path}`

To use the FieldSpace NN, We suggest to go to the root folder which include `MEssE`, git clone the project (more info [here](https://github.com/FREVA-CLINT/FieldSpaceNN))
```bash
git clone https://github.com/FREVA-CLINT/FieldSpaceNN.git
source activate ${desired_path}/messe_env/py_env/bin/activate  # activate the python environment
pip install -e .
```

## Run
First copy the prepared template `exp.atm_nwp_jsbach_xpp_r2b4` to the `run` folder under the build directory of ICON, then create the run script using `./make_runscripts --all`.

Run ICON with the plugin like so: `./scripts/run_icon_gpu.sh $ICON_BUILD_DIR $COMIN_PLUGIN_PATH $LEVANTE_ACCOUNT`


> `ICON_BUILD_DIR` is `${desired_path}/messe_env/build_dir/icon-model`  
> `COMIN_PLUGIN_PATH` is `$(pwd)/comin_plugin_pytorch/project_Z_pytorch.py`  
> `LEVANTE_ACCOUNT` is your levante project id

## Online interface 

```bash
./scripts/monitor.sh $LOG_FILE_PATH $local_host_port
```

Then access: **http://localhost:$local_host_port**
