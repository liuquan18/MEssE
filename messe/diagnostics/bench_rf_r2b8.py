"""GPU cost of one sample step of the RF plugin (reconstructor_forecaster/plugin.py) at R2B8, without ICON.

Splits the R2B8 grid into ``--ranks`` equal-area lat/lon boxes as a stand-in
for ICON's domain decomposition, assigns backbone cells to ranks as the plugin
does, builds the largest rank's patch from the real grid files, and times
`train_step` + `predict_increment` on a moving synthetic field. Not collected
by pytest. Run on a GPU node, from the project root:

    srun -A mh0033 -p gpu --gpus=1 --time=00:20:00 --mem=64G \\
        MEssE/build_nwp/messe_env/py_env/bin/python MEssE/messe/diagnostics/bench_rf_r2b8.py

``--replay-batch 0`` times the forecaster without latent replay; with replay,
the batch is full from step ``--replay-batch`` + 1 of the timed steps.
``--forcings`` sets the number of synthetic forcing fields that condition
both models (0 = none); as with the insolation in the plugin, the last one
conditions the forecaster only. ``--embed-dim`` is the size of each cell's
forcing embedding and of the time embedding.

``--neighbours`` is the configuration of ``config/fieldspace_rf_r2b8_neighbours.yaml``:
canonical cell order, the ghost cells of the patch, token overlap in both
models and two more forcing columns (the orientation). ``--device cpu`` runs
without a GPU, e.g. to count parameters.

``--reconstructor-attention``/``--forecaster-attention`` (global or local) and
``--token-len-time`` (forecaster time slots per token, default all) set the
attention layout. ``--memory-seconds`` turns on the key/value memory of the
last forecaster block (kv_memory.py; needs local forecaster attention and one
time token) with that span, sized for ``--replay-size``, and fills it with
random entries first, so that every timed step reads a full memory, as late in
a run. The configuration of ``config/fieldspace_rf_r2b8_kv_memory_local.yaml``:

    --neighbours --forcings 7 --n-context 2 --replay-size 1440 --reconstructor-attention local \\
    --forecaster-attention local --memory-seconds 86400
"""

import argparse
import time

import numpy as np
import torch

from messe.common.nested_grid import ghost_cells, load_global_mgrid, patch_mgrid
from messe.reconstructor_forecaster.model import OnlineReconstructorForecaster


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ranks", type=int, default=32)
    parser.add_argument("--levels", default="4,6,7,8")
    parser.add_argument("--latent-level", type=int, default=6)
    parser.add_argument("--steps", type=int, default=15, help="timed training steps")
    parser.add_argument("--replay-size", type=int, default=144)
    parser.add_argument("--replay-batch", type=int, default=10)
    parser.add_argument("--n-context", type=int, default=4)
    parser.add_argument("--forcings", type=int, default=4, help="plugin default: ts_wtr, sic, oromea, insolation")
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--inner-channels", type=int, default=None)
    parser.add_argument("--neighbours", action="store_true")
    parser.add_argument("--reconstructor-attention", default="global")
    parser.add_argument("--forecaster-attention", default="global")
    parser.add_argument("--token-len-time", type=int, default=None)
    parser.add_argument("--memory-seconds", type=float, default=0.0, help="key/value memory span; 0 = off")
    parser.add_argument("--memory-topk", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-dir", default="/pool/data/ICON/grids/mpim")
    args = parser.parse_args()
    device = torch.device(args.device)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    args.forcings += 2 * args.neighbours
    levels = [int(v) for v in args.levels.split(",")]
    refinement = 4 ** (levels[-1] - levels[0])

    t = time.perf_counter()
    global_mgrid = load_global_mgrid(args.grid_dir, levels[0], levels[-1], canonical=args.neighbours)
    lon, lat = global_mgrid[-1]["coords"].numpy().T
    # 4 equal-area latitude bands times ranks/4 longitude sectors.
    band = np.searchsorted([-0.5, 0.0, 0.5], np.sin(lat))
    sector = (np.degrees(lon) % 360 // (360 / (args.ranks // 4))).astype(int)
    rank_of_cell = band * (args.ranks // 4) + sector
    n_backbone = len(global_mgrid[0]["coords"])
    counts = np.stack([np.bincount(np.flatnonzero(rank_of_cell == r) // refinement, minlength=n_backbone) for r in range(args.ranks)])
    owner = counts.argmax(axis=0)
    per_rank = np.bincount(owner, minlength=args.ranks)
    owned = np.flatnonzero(owner == per_rank.argmax())
    ghosts = ghost_cells(owned, global_mgrid[0]["adjc"].numpy()) if args.neighbours else owned[:0]
    mgrids = patch_mgrid(global_mgrid, torch.from_numpy(np.concatenate([owned, ghosts])))
    coords = mgrids[-1]["coords"].to(device)
    print(f"setup {time.perf_counter() - t:.1f}s: backbone cells per rank {per_rank.min()}-{per_rank.max()}, "
          f"largest patch {len(owned)} owned + {len(ghosts)} ghost backbone cells = {len(coords)} R2B{levels[-1]} cells", flush=True)

    memory_steps = int(args.memory_seconds // 600)
    trainer = OnlineReconstructorForecaster(mgrids, levels, args.latent_level, nlev=1, n_context=args.n_context,
                                            n_forcings=args.forcings, reconstructor_forcings=range(args.forcings - 1),
                                            embed_dim=args.embed_dim, inner_channels=args.inner_channels,
                                            reconstructor_overlap=args.neighbours, forecaster_overlap=args.neighbours,
                                            n_owned=len(owned), device=device,
                                            replay_size=args.replay_size, replay_batch=args.replay_batch,
                                            reconstructor_attention=args.reconstructor_attention,
                                            forecaster_attention=args.forecaster_attention,
                                            forecaster_token_len_time=args.token_len_time,
                                            memory_size=args.n_context + args.replay_size + memory_steps if memory_steps else 0,
                                            memory_seconds=args.memory_seconds, memory_topk=args.memory_topk)
    print(f"{args.forcings} forcings, reconstructor "
          f"{sum(p.numel() for p in trainer.model.reconstructor.parameters()):,} / forecaster "
          f"{sum(p.numel() for p in trainer.model.forecaster.parameters()):,} parameters", flush=True)
    t0 = 1577836800.0
    memory = trainer.model.forecaster.memory
    if memory is not None:  # full, oldest first, stamped before the first timed window
        heads, cells = trainer.model.forecaster.memory_gate.shape[0], len(mgrids[0]["coords"])
        for j in reversed(range(memory.size)):  # 8 = the default n_head_channels
            memory.write(*torch.randn(2, heads, cells, 8, device=device), t0 - 600.0 * (j + 1))

    def forcing(k):  # (n, forcings)
        return torch.sin(10 * coords[:, 1:] * torch.arange(1, args.forcings + 1, device=coords.device) - 0.01 * k)

    for k in range(trainer.replay_buffer.n_context + args.steps):
        x = torch.sin(40 * coords[:, :1] - 0.02 * k) * torch.cos(20 * coords[:, 1:])  # (n, nlev=1)
        synchronize()
        t = time.perf_counter()
        stats = trainer.train_step(x, forcing(k), t0 + 600.0 * k)
        increment = trainer.predict_increment(t0 + 600.0 * (k + 1), forcing(k + 1))
        synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
        print(f"step {k}: {stats} step_s={time.perf_counter() - t:.2f} torch_peak_GiB={peak:.2f}", flush=True)
    assert torch.isfinite(increment).all()


if __name__ == "__main__":
    main()
