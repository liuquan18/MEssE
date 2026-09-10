"""MEssE: online-training framework for surrogate AI models trained
in-memory during live ICON simulations, via ComIn plugins.

Note: `comin_plugin_torch/` modules that are actual ComIn plugin scripts
(`gnn_plugin.py`, `unet_plugin.py`) are never imported as
`MEssE.comin_plugin_torch.gnn_plugin` — COMIN execs those files directly,
bypassing normal package import machinery, so this `__init__.py` never runs
as a side effect of that. It exists so `MEssE.utils.*` (and, for tests,
`MEssE.comin_plugin_torch.*`) can be imported as a regular package.
"""
