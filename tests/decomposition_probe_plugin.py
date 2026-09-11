"""Standalone diagnostic COMIN plugin: print ICON's domain decomposition
and the resulting FieldSpaceNN two-zoom multi-grid for every rank, with no
training, no dry-run, no GPU/CUDA needed at all (the multi-grid is built
on CPU purely to inspect its shapes/stats).

Submit like any other plugin via run_icon_gpu.sh:
    ./run_icon_gpu.sh <ICON_BUILD_DIR> \
        .../MEssE/tests/decomposition_probe_plugin.py <ACCOUNT> <EXPNAME>
then read the results straight out of the job's log file -- everything
this script prints happens at plugin-load time (module level, before any
simulation timestep), so the job can be cancelled (`scancel`) as soon as
the printout appears; there is no need to let it run its full wall-clock
allocation.

Not a pytest test (imports `comin`, only importable inside a live
ICON/COMIN process) -- lives in MEssE/tests/ for colocation with the rest
of the FieldSpaceNN-plugin test suite only. Deliberately NOT named
`test_*.py`/`*_test.py` so pytest's default discovery does not try (and
fail) to collect it.
"""

import os
import sys

import comin
import numpy as np
import torch
from mpi4py import MPI

try:
    _PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ may be undefined when COMIN loads the plugin via exec().
    _PLUGIN_DIR = os.environ.get("MESSE_PLUGIN_DIR", os.getcwd())

# This file lives in MEssE/tests/, which is the SAME depth under the
# project root as the real plugins' MEssE/comin_plugin_torch/ (both one
# level under MEssE/) -- so _PROJECT_ROOT needs the same two dirname()
# calls the real plugins use (tests/ -> MEssE/ -> Project_week_global/),
# not just one (an earlier version of this file got this wrong: one
# dirname() lands on MEssE/ itself, not the project root, so
# "MEssE"/comin_plugin_torch got joined onto MEssE/ a second time and
# icon_mgrid_utils was never actually found -- confirmed by
# ModuleNotFoundError on a real run, job 27402376).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PLUGIN_DIR))
_COMIN_PLUGIN_TORCH_DIR = os.path.join(_PROJECT_ROOT, "MEssE", "comin_plugin_torch")
for _p_path in (_PROJECT_ROOT, _COMIN_PLUGIN_TORCH_DIR):
    if _p_path not in sys.path:
        sys.path.insert(0, _p_path)

from icon_mgrid_utils import build_local_mgrid, load_coarse_grid, load_parent_index

DOMAIN_ID = int(os.environ.get("MESSE_DOMAIN_ID", "1"))
COARSE_GRID_PATH = os.environ.get(
    "MESSE_FS_COARSE_GRID_PATH",
    "/pool/data/ICON/grids/public/edzw/icon_grid_0011_R02B03_R.nc",
)
FINE_GRID_PATH = os.environ.get(
    "MESSE_FS_FINE_GRID_PATH",
    "/pool/data/ICON/grids/public/edzw/icon_grid_0012_R02B04_G.nc",
)


def _log(msg: str) -> None:
    # comin.print_info only actually prints on rank 0 (confirmed
    # empirically on a real run, job 27394423 -- every other rank's
    # comin.print_info call was silently dropped). Plain stderr print
    # instead, so every rank's line survives; srun still prefixes each
    # line with the MPI task number regardless of stdout vs stderr.
    print(msg, file=sys.stderr, flush=True)


comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
world_size = comm.Get_size()
rank = comm.Get_rank()
local_rank = int(os.environ.get("SLURM_LOCALID", "-1"))

# Same classification icon_online_helper.setup_mpi_dist uses for the real
# plugins: this runscript launches 5 MPI tasks per node (4 GPU/compute +
# 1 dedicated I/O rank, per parallel_nml's num_io_procs=1), and
# SLURM_LOCALID < gpus_per_node is how those two roles are told apart --
# there's no other signal. The I/O rank's `domain.cells` decomposition is
# not expected to look like a normal compute patch (may well own zero
# cells), so it's labeled explicitly here rather than silently averaged in
# with the 4 real GPU domains.
GPUS_PER_NODE = 4
is_gpu_rank = 0 <= local_rank < GPUS_PER_NODE

glob = comin.descrdata_get_global()
domain = comin.descrdata_get_domain(DOMAIN_ID)

nproma = int(glob.nproma)
n_valid = int(domain.cells.ncells)  # owned + halo, excludes unused nproma padding
n_blocks = int(domain.cells.nblks)

decomp = np.ravel(np.asarray(domain.cells.decomp_domain)).astype(np.int64)[:n_valid]
owned_mask = decomp == 0
n_owned = int(owned_mask.sum())
n_halo = n_valid - n_owned

clon_deg = np.degrees(np.ravel(np.asarray(domain.cells.clon))[:n_valid])
clat_deg = np.degrees(np.ravel(np.asarray(domain.cells.clat))[:n_valid])

grid_filename = getattr(domain, "grid_filename", "<unavailable>")

_log(
    f"[rank={rank} local_rank={local_rank} is_gpu_rank={is_gpu_rank}] RAW DECOMPOSITION: "
    f"grid_filename={grid_filename} nproma={nproma} nblks={n_blocks} "
    f"n_valid(owned+halo)={n_valid} n_owned={n_owned} n_halo={n_halo} "
    + (
        f"owned_lon_deg=[{clon_deg[owned_mask].min():.1f},{clon_deg[owned_mask].max():.1f}] "
        f"owned_lat_deg=[{clat_deg[owned_mask].min():.1f},{clat_deg[owned_mask].max():.1f}]"
        if n_owned > 0
        else "owned_lon_deg=[] owned_lat_deg=[] (no owned cells on this rank)"
    )
)

# Global sanity check: ownership should be an exact, non-overlapping
# partition of the whole grid. Reported two ways: summed over just the 4
# GPU/compute ranks (the ones that actually run FieldSpaceNN training) and
# summed over all `world_size` ranks (GPU + I/O), since it's not yet known
# without running this whether ICON's own decomposition gives the I/O rank
# a (nonzero) slice of owned cells too -- both sums are printed so that
# question is answered by the run itself, not assumed.
total_owned_gpu = comm.allreduce(n_owned if is_gpu_rank else 0, op=MPI.SUM)
total_owned_all = comm.allreduce(n_owned, op=MPI.SUM)
if rank == 0:
    _log(
        f"[GLOBAL] total_owned summed over the 4 GPU ranks = {total_owned_gpu}; "
        f"summed over all {world_size} ranks (GPU + I/O) = {total_owned_all}"
    )

# --- Two-zoom multi-grid construction (coarse R2B3 + fine R2B4) ---
# Only on the 4 GPU/compute ranks, matching the real fieldspace_plugin.py's
# `if has_gpu:` gate exactly -- the I/O rank never builds an mgrid in
# production, so skip it here too rather than exercising a code path that
# never actually runs for real. CPU only: this script never trains, so
# there's no need for CUDA/cupy at all, just the grid-construction logic.
n_owned_kept = 0
if is_gpu_rank:
    try:
        coarse_grid = load_coarse_grid(COARSE_GRID_PATH)
        parent_index_global = load_parent_index(FINE_GRID_PATH)
        n_global_fine = int(parent_index_global.shape[0])

        mgrid = build_local_mgrid(
            domain,
            nproma=nproma,
            coarse_grid=coarse_grid,
            parent_index_global=parent_index_global,
            device=torch.device("cpu"),
        )
        n_owned_kept = int(mgrid.fine_compact_owned_mask.sum().item())

        _log(
            f"[rank={rank}] MGRID: n_global_fine={n_global_fine} "
            f"coarse_grid.n_cells={coarse_grid.n_cells} "
            f"n_fine_kept={mgrid.n_fine_kept} n_coarse={mgrid.n_coarse} "
            f"kept_fraction(of owned+halo)={mgrid.n_fine_kept / max(n_valid, 1):.3f} "
            f"n_owned_kept={n_owned_kept} "
            f"owned_kept_fraction={n_owned_kept / max(n_owned, 1):.3f} "
            f"group_sizes={mgrid.parent_group_size_histogram}"
        )
    except Exception as e:
        _log(f"[rank={rank}] MGRID construction FAILED: {type(e).__name__}: {e}")
else:
    _log(f"[rank={rank}] MGRID: skipped (I/O rank, matches fieldspace_plugin.py's has_gpu gate)")

# Sum only over the 4 GPU ranks -- n_owned_kept is 0 (never computed) on
# the I/O rank by construction above.
total_owned_kept_gpu = comm.allreduce(n_owned_kept, op=MPI.SUM)
if rank == 0:
    _log(
        f"[GLOBAL] total_owned_kept summed over the 4 GPU ranks = {total_owned_kept_gpu} "
        f"/ total_owned (GPU ranks) = {total_owned_gpu} -> "
        f"global_owned_kept_fraction={total_owned_kept_gpu / max(total_owned_gpu, 1):.4f}"
    )

_log(f"[rank={rank}] decomposition_probe_plugin done.")


@comin.register_callback(comin.EP_DESTRUCTOR)
def destructor():
    pass
