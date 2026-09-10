"""Shared, model-agnostic helpers for MEssE's online-training plugins:
`icon_online_helper.py` (COMIN<->tensor/MPI/checkpoint I/O, online
normalization, and per-timestep training-example scheduling — no YAC/HEALPix)
and `healpix_grids.py` (the YAC+HEALPix-specific counterpart, used only by
the interpolation-based UNet plugin).
"""
