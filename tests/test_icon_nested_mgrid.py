"""Tests for icon_nested_mgrid.py: rank patches and the cell exchange.

Synthetic nested grids in FieldSpaceNN's ``mgrids`` format. The exchange runs
on an in-process communicator (one thread per rank) with mpi4py's
Alltoall/Alltoallv buffer semantics, so no MPI job is needed.
"""

import threading

import numpy as np
import torch

from icon_nested_mgrid import CellExchange, patch_mgrid

N_BASE = 12  # backbone cells of the synthetic globe


def synthetic_global_mgrid(n_levels=3):
    """FieldSpaceNN-style ``mgrids`` of a nested ring-like globe: neighbors
    c+1, c-1, c+n/2, tiled into 9 columns like `icon_neighbor_cell_index_to_adjc`."""
    mgrids = []
    for zoom in range(n_levels):
        n = N_BASE * 4**zoom
        c = torch.arange(n)
        neighbors = torch.stack([(c + 1) % n, (c - 1) % n, (c + n // 2) % n], dim=1)
        mgrids.append(
            {
                "coords": torch.stack([torch.linspace(-3, 3, n), torch.zeros(n)], dim=-1),
                "adjc": torch.cat([c[:, None], neighbors[:, [0, 1, 2, 0, 1, 2, 0, 1]]], dim=1),
                "adjc_mask": torch.zeros(n, 9, dtype=torch.bool),
                "zoom": zoom,
            }
        )
    return mgrids


def test_patch_is_the_nested_descendants_with_local_adjacency():
    global_mgrid = synthetic_global_mgrid()
    backbone = torch.tensor([2, 3, 7])
    patch = patch_mgrid(global_mgrid, backbone)
    for zoom, (grid, local) in enumerate(zip(global_mgrid, patch)):
        ids = (backbone[:, None] * 4**zoom + torch.arange(4**zoom)).flatten()
        assert torch.equal(local["coords"], grid["coords"][ids])
        global_adjc = grid["adjc"][ids]
        inside = torch.isin(global_adjc, ids)
        # Neighbors inside the patch are the global ones, renumbered to patch rows.
        assert torch.equal(ids[local["adjc"]][inside], global_adjc[inside])
        # Neighbors outside fall back to the cell itself and are masked, except
        # in the polar-fix columns 2 and 6.
        rows = torch.arange(len(ids))[:, None].expand(-1, 9)
        assert torch.equal(local["adjc"][~inside], rows[~inside])
        columns = [0, 1, 3, 4, 5, 7, 8]
        assert torch.equal(local["adjc_mask"][:, columns], ~inside[:, columns])
        assert not local["adjc_mask"][:, [2, 6]].any()


class _Hub:
    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.deposits = {}


class _ThreadComm:
    """One rank of an in-process communicator with mpi4py's buffer semantics."""

    def __init__(self, hub, rank):
        self.hub, self.rank, self.calls = hub, rank, 0

    def Get_size(self):
        return self.hub.size

    def Get_rank(self):
        return self.rank

    def _rendezvous(self, payload):
        key = self.calls
        self.calls += 1
        self.hub.deposits[(key, self.rank)] = payload
        self.hub.barrier.wait(timeout=30)
        return [self.hub.deposits[(key, r)] for r in range(self.hub.size)]

    def Alltoall(self, sendbuf, recvbuf):
        sends = self._rendezvous(np.array(sendbuf, copy=True))
        for src in range(self.hub.size):
            recvbuf[src] = sends[src][self.rank]

    def Alltoallv(self, send, recv):
        sendbuf, (scounts, sdispls) = send
        recvbuf, (rcounts, rdispls) = recv
        sends = self._rendezvous((sendbuf.ravel().copy(), list(scounts), list(sdispls)))
        flat = recvbuf.reshape(-1)
        for src, (buf, counts, displs) in enumerate(sends):
            n = counts[self.rank]
            assert n == rcounts[src]
            flat[rdispls[src]: rdispls[src] + n] = buf[displs[self.rank]: displs[self.rank] + n]


def _run_ranks(size, fn):
    hub = _Hub(size)
    results, errors = [None] * size, []

    def target(rank):
        try:
            results[rank] = fn(_ThreadComm(hub, rank))
        except BaseException as e:  # noqa: BLE001 - re-raised below
            errors.append(e)
            hub.barrier.abort()

    threads = [threading.Thread(target=target, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


def test_exchange_routes_every_fine_cell_to_one_patch_and_back():
    size, refinement, nlev = 3, 16, 2
    n_fine = N_BASE * refinement
    rng = np.random.default_rng(0)
    rank_of_cell = rng.integers(0, size, n_fine)  # ownership ignores the hierarchy
    field = rng.normal(size=(n_fine, nlev)).astype(np.float32)

    # Local ICON view per rank: owned cells in a shuffled local order, plus
    # halo rows (NaN) that must never be sent.
    local = []
    for r in range(size):
        owned_glb = np.flatnonzero(rank_of_cell == r)
        halo_glb = rng.choice(np.flatnonzero(rank_of_cell != r), 5, replace=False)
        glb = rng.permutation(np.concatenate([owned_glb, halo_glb]))
        values = field[glb].copy()
        values[np.isin(glb, halo_glb)] = np.nan
        local.append((glb, values, np.flatnonzero(np.isin(glb, owned_glb))))

    # The plugin's owner rule: the rank owning most fine cells of a backbone cell.
    counts = np.stack([np.bincount(glb[owned] // refinement, minlength=N_BASE) for glb, _, owned in local])
    owner = counts.argmax(axis=0)

    def rank_fn(comm):
        glb, values, owned_local = local[comm.Get_rank()]
        backbone = np.flatnonzero(owner == comm.Get_rank())
        exchange = CellExchange.build(comm, owned_local, glb[owned_local], owner, backbone, refinement)
        patch = exchange.to_patch(comm, values)
        return backbone, patch, exchange, exchange.from_patch(comm, patch * 10.0)

    covered = []
    for r, (backbone, patch, exchange, back) in enumerate(_run_ranks(size, rank_fn)):
        ids = (backbone[:, None] * refinement + np.arange(refinement)).ravel()
        assert np.array_equal(patch, field[ids])  # nested order, no halo values
        covered.append(ids)
        glb, _, owned_local = local[r]
        assert np.array_equal(np.sort(exchange.send_ids), owned_local)
        assert np.array_equal(back, field[glb[exchange.send_ids]] * 10.0)
    assert np.array_equal(np.sort(np.concatenate(covered)), np.arange(n_fine))
