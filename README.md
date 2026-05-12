# Model Essence Extractor (MEssE)
High-frequency output for km-scale simulations is too big to be stored, but AI emulators precisely rely on these data for training. To address this dilemma, we introduce MEssE --- a framework that trians a surrogate AI model during ICON simulations using in-memory data. 

# Structure
```text
.
├── comin_plugin_JAX                    # project in JAX *under development*
├── comin_plugin_Pytorch                # in Pytorch 
│   ├── facecnn_online.py               # simple example of CNN using HEALPix grid to test
│   ├── fieldspacenn_online.py          # configuration of [FieldSpaceNN](https://github.com/FREVA-CLINT/FieldSpaceNN) ready for online training
│   └── project_Z_pytorch.py            # data preparation and online training
├── monitors                            # *under development*
└── scripts                             # scripts 
    ├── build_env_gpu.sh                # building ICON, ComIn, YAC and preparing python env
    ├── run_icon_gpu.sh                 # running ICON with GPU
    └── exp.aes_amip_messe_test.run     # prepared running scripts 
```
# Workflow

## Build

Prepare environment using `./scripts/build_env_gpu.sh ${desired_path}`
On successful completion of the script, a py_venv is created under `${desired_path}/messe_env/py_env`,
and ICON is built in `${desired_path}/messe_env/build_dir/icon-model`

To use the FieldSpace NN, We suggest to go to the root folder which include `MEssE`, git clone the project (more info [here](https://github.com/FREVA-CLINT/FieldSpaceNN))
```bash
git clone https://github.com/FREVA-CLINT/FieldSpaceNN.git
source activate ${desired_path}/messe_env/py_env/bin/activate  # activate the python environment
pip install -e .
```

## Run
Run ICON with the plugin like so: `./scripts/run_icon_gpu.sh $ICON_BUILD_DIR $COMIN_PLUGIN_PATH $LEVANTE_ACCOUNT`

```bash
`ICON_BUILD_DIR` can be found under `${desired_path}/build_dir`
`COMIN_PLUGIN_PATH` is `$(pwd)/comin_plugin_pytorch/project_Z_pytorch.py`
`LEVANTE_ACCOUNT` is your levante project id
```

## Online interface (unavailable for now)

```bash
cd $(pwd)/monitor
./start_monitor.sh 5000 #(any port number)
```

Then access: **http://localhost:5000**
