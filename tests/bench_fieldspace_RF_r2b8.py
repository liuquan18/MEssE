"""GPU cost of one sample step of fieldspace_RF_plugin.py at R2B8, without ICON.

Splits the R2B8 grid into ``--ranks`` equal-area lat/lon boxes as a stand-in
for ICON's domain decomposition, assigns backbone cells to ranks as the plugin
does, builds the largest rank's patch from the real grid files, and times
`train_step` + `predict_increment` on a moving synthetic field. Not collected
by pytest. Run on a GPU node, from the project root:

    srun -A mh0033 -p gpu --gpus=1 --time=00:20:00 --mem=64G \\
        MEssE/build_nwp/messe_env/py_env/bin/python MEssE/tests/bench_fieldspace_RF_r2b8.py
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(os.path.dirname(_HERE)), os.path.join(os.path.dirname(_HERE), "comin_plugin_torch")]

from fieldspace_RF_online import OnlineReconstructorForecaster  # noqa: E402
from icon_nested_mgrid import load_global_mgrid, patch_mgrid  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ranks", type=int, default=32)
    parser.add_argument("--levels", default="4,6,7,8")
    parser.add_argument("--latent-level", type=int, default=6)
    parser.add_argument("--steps", type=int, default=5, help="timed training steps")
    parser.add_argument("--grid-dir", default="/pool/data/ICON/grids/mpim")
    args = parser.parse_args()
    levels = [int(v) for v in args.levels.split(",")]
    refinement = 4 ** (levels[-1] - levels[0])

    t = time.perf_counter()
    global_mgrid = load_global_mgrid(args.grid_dir, levels[0], levels[-1])
    lon, lat = global_mgrid[-1]["coords"].numpy().T
    # 4 equal-area latitude bands times ranks/4 longitude sectors.
    band = np.searchsorted([-0.5, 0.0, 0.5], np.sin(lat))
    sector = (np.degrees(lon) % 360 // (360 / (args.ranks // 4))).astype(int)
    rank_of_cell = band * (args.ranks // 4) + sector
    n_backbone = len(global_mgrid[0]["coords"])
    counts = np.stack([np.bincount(np.flatnonzero(rank_of_cell == r) // refinement, minlength=n_backbone) for r in range(args.ranks)])
    owner = counts.argmax(axis=0)
    per_rank = np.bincount(owner, minlength=args.ranks)
    backbone = torch.from_numpy(np.flatnonzero(owner == per_rank.argmax()))
    mgrids = patch_mgrid(global_mgrid, backbone)
    coords = mgrids[-1]["coords"].cuda()
    print(f"setup {time.perf_counter() - t:.1f}s: backbone cells per rank {per_rank.min()}-{per_rank.max()}, "
          f"largest patch {len(coords)} R2B{levels[-1]} cells", flush=True)

    trainer = OnlineReconstructorForecaster(mgrids, levels, args.latent_level, nlev=1)
    t0 = 1577836800.0
    for k in range(trainer.history.maxlen + args.steps):
        x = (torch.sin(40 * coords[:, :1] - 0.02 * k) * torch.cos(20 * coords[:, 1:]))
        torch.cuda.synchronize()
        t = time.perf_counter()
        stats = trainer.train_step(x, t0 + 600.0 * k)
        increment = trainer.predict_increment(t0 + 600.0 * (k + 1))
        torch.cuda.synchronize()
        print(f"step {k}: {stats} step_s={time.perf_counter() - t:.2f} "
              f"torch_peak_GiB={torch.cuda.max_memory_allocated() / 2**30:.2f}", flush=True)
    assert torch.isfinite(increment).all()


if __name__ == "__main__":
    main()
