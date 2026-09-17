"""GPU cost of one online training step of fieldspace_AE_plugin.py at R2B8,
without ICON.

Builds one rank's local patch the way COMIN would hand it over (owned cells
plus two halo rings, in shuffled local order) from the real R2B8 grid file,
builds the multi-grid with the real R2B7 parent grid, and times
`OnlineFieldSpaceAETrainer.train_step`/`predict` with a synthetic 2D field.
The default box (0-30 N, 0-90 E) is 1/16 of the globe, i.e. one rank of 16.
Not collected by pytest (no ``test_`` prefix). Run on a GPU node:

    srun -A mh0033 -p gpu --gpus=1 --time=00:20:00 --mem=64G \\
        MEssE/build_nwp/messe_env/py_env/bin/python MEssE/tests/bench_fieldspace_AE_r2b8.py
"""

import argparse
import math
import os
import sys
import time
from types import SimpleNamespace

import netCDF4
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(os.path.dirname(_HERE)), os.path.join(os.path.dirname(_HERE), "comin_plugin_torch")]

from fieldspace_AE_online import OnlineFieldSpaceAETrainer  # noqa: E402
from icon_mgrid_utils import (  # noqa: E402
    build_local_mgrid,
    check_grid_files,
    load_coarse_grid,
    load_parent_index,
)

FINE = "/pool/data/ICON/grids/mpim/Earth_IcosS_0010km.nc"
COARSE = "/pool/data/ICON/grids/mpim/Earth_IcosS_0020km.nc"


def fake_rank(lat_range, lon_range, seed=0):
    """A COMIN-like `domain` for the cells in a lat/lon box (degrees) plus two
    edge-neighbor halo rings, with nblks=1."""
    with netCDF4.Dataset(FINE) as ds:
        lon = np.asarray(ds["lon_cell_centre"][:], dtype=np.float64)
        lat = np.asarray(ds["lat_cell_centre"][:], dtype=np.float64)
        nbr = np.asarray(ds["neighbor_cell_index"][:]).T.astype(np.int64) - 1
    if np.abs(lat).max() > 4:  # stored in degrees
        lon, lat = np.radians(lon), np.radians(lat)
    lon_deg, lat_deg = np.degrees(lon) % 360, np.degrees(lat)
    owned = np.where(
        (lat_deg >= lat_range[0]) & (lat_deg < lat_range[1])
        & (lon_deg >= lon_range[0]) & (lon_deg < lon_range[1])
    )[0]

    level = np.full(lon.size, -1, dtype=np.int64)
    level[owned] = 0
    frontier = owned
    for ring in (1, 2):
        cand = np.unique(nbr[frontier].ravel())
        frontier = cand[level[cand] < 0]
        level[frontier] = ring

    glb0 = np.where(level >= 0)[0]
    glb0 = glb0[np.random.default_rng(seed).permutation(glb0.size)]
    n_valid = glb0.size
    nproma = n_valid + 17
    to_local = np.full(lon.size, -1, dtype=np.int64)
    to_local[glb0] = np.arange(n_valid)
    local_nbr = to_local[nbr[glb0]]  # (n_valid, 3), -1 if not on this rank

    neighbor_idx = np.zeros((nproma, 1, 3), dtype=np.int32)
    neighbor_idx[:n_valid, 0] = np.where(local_nbr >= 0, local_nbr + 1, 0)
    neighbor_blk = (neighbor_idx > 0).astype(np.int32)

    def column(values, fill):
        out = np.full((nproma, 1), fill, dtype=values.dtype)
        out[:n_valid, 0] = values
        return out

    cells = SimpleNamespace(
        ncells=n_valid,
        nblks=1,
        ncells_global=lon.size,
        clon=column(lon[glb0], 0.0),
        clat=column(lat[glb0], 0.0),
        decomp_domain=column(level[glb0].astype(np.int32), -1),
        neighbor_idx=neighbor_idx,
        neighbor_blk=neighbor_blk,
        glb_index=(glb0 + 1).astype(np.int32),
    )
    return SimpleNamespace(cells=cells), nproma


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lat", type=float, nargs=2, default=[0.0, 30.0])
    parser.add_argument("--lon", type=float, nargs=2, default=[0.0, 90.0])
    parser.add_argument("--steps", type=int, default=5, help="timed training steps")
    parser.add_argument("--n-history", type=int, default=4)
    parser.add_argument("--device", default="cuda:0", help="cpu only checks the script on a small --lat/--lon box")
    args = parser.parse_args()
    device = torch.device(args.device)
    on_gpu = device.type == "cuda"

    def sync():
        if on_gpu:
            torch.cuda.synchronize()

    t = time.perf_counter()
    coarse = load_coarse_grid(COARSE)
    parent = load_parent_index(FINE)
    domain, nproma = fake_rank(args.lat, args.lon)
    check_grid_files(parent, coarse, domain.cells.ncells_global)
    mgrid = build_local_mgrid(domain, nproma=nproma, coarse_grid=coarse, parent_index_global=parent, device=device)
    print(
        f"setup {time.perf_counter() - t:.1f}s: n_valid={domain.cells.ncells} n_coarse={mgrid.n_coarse} "
        f"n_fine_kept={mgrid.n_fine_kept}",
        flush=True,
    )

    t = time.perf_counter()
    trainer = OnlineFieldSpaceAETrainer(
        nlev=1, mgrid=mgrid, n_history=args.n_history, device=device, use_ddp=False, log_fn=print, rank=0
    )
    print(f"trainer built in {time.perf_counter() - t:.1f}s", flush=True)

    lon, lat = mgrid.fine_coords[:, :1], mgrid.fine_coords[:, 1:]
    t0 = 1577836800.0  # 2020-01-01T00:00Z
    for k in range(trainer.reservoir.capacity + args.steps):
        x = torch.sin(20 * lon - 0.05 * k) * torch.cos(10 * lat)
        snapshot = trainer.prepare_snapshot(x, t0 + 60.0 * k)
        sync()
        t = time.perf_counter()
        result = trainer.train_step(snapshot)
        sync()
        t_train = time.perf_counter() - t
        t = time.perf_counter()
        pred = trainer.predict(n_steps=1)
        sync()
        t_pred = time.perf_counter() - t
        peak_gib = torch.cuda.max_memory_allocated() / 2**30 if on_gpu else float("nan")
        print(
            f"step {k}: skipped={result['skipped']} loss={result['loss']:.4f} train_s={t_train:.2f} "
            f"predict_s={t_pred:.2f} torch_peak_GiB={peak_gib:.2f}",
            flush=True,
        )
    assert pred is not None and torch.isfinite(pred).all() and math.isfinite(result["loss"])


if __name__ == "__main__":
    main()
