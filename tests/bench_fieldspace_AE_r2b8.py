"""GPU cost of one online training step of fieldspace_AE_plugin.py at R2B8,
without ICON.

Splits the R2B8 grid into ``--ranks`` equal-area lat/lon boxes as a stand-in
for ICON's domain decomposition, assigns backbone cells to ranks the way the
plugin does, builds the largest rank's nested patch from the real grid files,
and times `OnlineFieldSpaceAETrainer.train_step`/`predict` on a synthetic 2D
field. Not collected by pytest (no ``test_`` prefix). Run on a GPU node, from
the project root:

    srun -A mh0033 -p gpu --gpus=1 --time=00:20:00 --mem=64G \\
        MEssE/build_nwp/messe_env/py_env/bin/python MEssE/tests/bench_fieldspace_AE_r2b8.py
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(os.path.dirname(_HERE)), os.path.join(os.path.dirname(_HERE), "comin_plugin_torch")]

from fieldspace_AE_online import OnlineFieldSpaceAETrainer  # noqa: E402
from icon_nested_mgrid import (  # noqa: E402
    assign_backbone_owners,
    backbone_counts,
    build_nested_mgrid,
    load_level_grids,
    mpim_grid_paths,
)


def box_ranks(lon: np.ndarray, lat: np.ndarray, n_ranks: int) -> np.ndarray:
    """Rank per cell: 4 equal-area latitude bands times n_ranks/4 longitude sectors."""
    if n_ranks % 4:
        raise ValueError("--ranks must be a multiple of 4")
    band = np.clip(np.searchsorted([-0.5, 0.0, 0.5], np.sin(lat), side="right"), 0, 3)
    sector = ((np.degrees(lon) % 360) // (360 / (n_ranks // 4))).astype(np.int64) % (n_ranks // 4)
    return band * (n_ranks // 4) + sector


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ranks", type=int, default=32)
    parser.add_argument("--levels", default="4,6,7,8")
    parser.add_argument("--latent-level", type=int, default=6)
    parser.add_argument("--steps", type=int, default=5, help="timed training steps")
    parser.add_argument("--n-history", type=int, default=4)
    parser.add_argument("--grid-dir", default="/pool/data/ICON/grids/mpim")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    levels = sorted(int(v) for v in args.levels.split(","))
    device = torch.device(args.device)
    on_gpu = device.type == "cuda"

    def sync():
        if on_gpu:
            torch.cuda.synchronize()

    t = time.perf_counter()
    grids = load_level_grids(mpim_grid_paths(args.grid_dir, levels[0], levels[-1]))
    fine = grids[levels[-1]]
    refinement = 4 ** (levels[-1] - levels[0])
    rank_of_cell = box_ranks(fine.lon, fine.lat, args.ranks)
    counts = np.stack(
        [backbone_counts(np.flatnonzero(rank_of_cell == r), grids[levels[0]].n_cells, refinement) for r in range(args.ranks)]
    )
    owner = assign_backbone_owners(counts, refinement)
    per_rank = np.bincount(owner, minlength=args.ranks)
    rank = int(np.argmax(per_rank))
    mgrid = build_nested_mgrid(grids, np.flatnonzero(owner == rank), device=device)
    print(
        f"setup {time.perf_counter() - t:.1f}s: backbone cells per rank {per_rank.min()}-{per_rank.max()}, "
        f"largest patch rank {rank}: n_fine={mgrid.n_fine}",
        flush=True,
    )

    trainer = OnlineFieldSpaceAETrainer(
        nlev=1, mgrid=mgrid, levels=levels, latent_level=args.latent_level, n_history=args.n_history,
        device=device, use_ddp=False, log_fn=print, rank=0,
    )
    lon, lat = mgrid.coords[-1][:, :1], mgrid.coords[-1][:, 1:]
    t0 = 1577836800.0  # 2020-01-01T00:00Z
    for k in range(trainer.reservoir.capacity + args.steps):
        x = torch.sin(40 * lon - 0.02 * k) * torch.cos(20 * lat)
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
