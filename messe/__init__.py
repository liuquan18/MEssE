"""MEssE: AI models trained online, in memory, inside a running ICON simulation (ComIn plugins).

- common/                    used by more than one model: ComIn <-> tensors, MPI setup,
                             normalization, checkpoints, rank patches of the nested grids;
- reconstructor_forecaster/  the main model (RF): plugin, model, config, kv memory, forcings;
- other_models/              autoencoder, gnn, unet, jax;
- diagnostics/               decomposition probe plugin and the RF GPU benchmark.

Each `plugin.py` is run by ComIn by path (`plugin_list(1)%options`), not imported. They
import everything else as `messe.*`, which works because the package is installed in the
Python env (`pip install -e MEssE`).
"""
