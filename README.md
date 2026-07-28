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
│   ├── unet_plugin.py                  # data preparation and online training
│   ├── graph_utils.py                  # ICON native-grid local (per-rank) patch graph builder
│   ├── gnn_online.py                   # GNN model (encoder-processor-decoder) + DDP trainer
│   └── gnn_plugin.py                   # GNN plugin: trains directly on the ICON native grid
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

## GNN on the ICON native grid (`gnn_plugin.py`)

`comin_plugin_torch/gnn_plugin.py` is an alternative to the UNet/HEALPix
plugin that trains a graph neural network **directly on the ICON native
(triangular) grid**, avoiding YAC and HEALPix interpolation entirely.

Design:
* **Patch = one rank's ICON domain-decomposed cells.** Each GPU already
  owns a contiguous region of the global mesh (its "owned" prognostic
  cells plus a halo ring shared by ICON's own MPI halo exchange). We treat
  that region as one training sample/graph, so the GPU layout and DDP
  world are inherited unmodified from ICON — no separate patch scheduling
  or interpolation step is required. This is also intended to generalize
  to km-scale resolution, where a full-globe sample would be far too
  large; patch-based, rollout-based training keeps the per-step working
  set bounded regardless of resolution.
* **Graph construction** (`graph_utils.py`): nodes are all local cells
  returned by COMIN for a rank (owned + halo), edges come from ICON's
  native cell-to-cell adjacency (`domain.cells.neighbor_idx/neighbor_blk`,
  up to 3 neighbors per triangle), added in both directions with
  self-loops. Halo cells are kept as message-passing-only nodes (via
  `domain.cells.decomp_domain`) so boundary cells still receive correct
  neighbor information; the training/inference loss is masked to owned
  cells only so halo duplication across ranks never inflates the loss.
* **Model** (`gnn_online.py`): a small encoder-processor-decoder GNN
  (inspired by [anemoi](https://github.com/ecmwf/anemoi)/GraphCast) that
  predicts a per-node tendency. Message passing is implemented directly in
  PyTorch (no new dependency on `torch_geometric`/`anemoi-models`) since
  the per-rank patch graph is small; either library could be dropped in
  later without changing the plugin-facing API.
* **Training**: rollout-based — the model is unrolled autoregressively for
  `MESSE_GNN_ROLLOUT_STEPS` steps (default 4) with its own predictions fed
  back as input, and the summed/averaged loss over the rollout is
  backpropagated once per window, mirroring the UNet plugin's
  DDP + checkpoint/rollback conventions.

Key environment variables: `MESSE_ICON_VAR`, `MESSE_GNN_ROLLOUT_STEPS`,
`MESSE_GNN_HIDDEN_DIM`, `MESSE_GNN_PROCESSOR_LAYERS`, `MESSE_GNN_LR`.

