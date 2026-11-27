"""MEssE: AI models trained online, in memory, inside a running ICON simulation (ComIn plugins).

- common_utils/             ComIn <-> tensors, MPI setup, normalization, checkpoints,
                             rank patches of the nested grids;
- reconstructor_forecaster/  the model (RF): plugin, model, config, kv memory, forcings.

Each `plugin.py` is run by ComIn by path (`plugin_list(1)%options`), not imported. They
import everything else as `messe.*`, which works because the package is installed in the
Python env (`pip install -e MEssE`).
"""
