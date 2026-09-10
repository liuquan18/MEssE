"""ComIn plugin scripts and their model-specific training code.

`gnn_plugin.py` and `unet_plugin.py` are ComIn plugins: COMIN execs those
files directly (not `import`), so nothing in this `__init__.py` runs as a
side effect of that, and nothing in this package should assume it does.
`graph_utils.py`, `gnn_online.py`, `unet_online.py` are plain, side-effect-free
modules imported by bare name (flat, not dotted) both by the plugins and by
`MEssE/tests/`, matching how COMIN's exec-based loading works.
"""
