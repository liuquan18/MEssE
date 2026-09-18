"""Unit tests for icon_nested_mgrid.py: grid loading and nesting checks,
backbone ownership, nested patches, and the cell exchange between ranks.

Synthetic nested grids, no comin, no GPU. The exchange runs on an in-process
communicator (one thread per rank) with mpi4py's Alltoall/Alltoallv buffer
semantics, so no MPI job is needed.
"""

import threading

import netCDF4
import numpy as np
import pytest
import torch

from icon_nested_mgrid import (
    CellExchange,
    LevelGrid,
    assign_backbone_owners,
    backbone_counts,
    build_nested_mgrid,
    check_cell_centers,
    check_index_nesting,
    load_level_grids,
    patch_cell_ids,
    patch_positions,
)

N_BASE = 12  # backbone cells of the synthetic global grid


def _neighbors(n):
    c = np.arange(n)
    return np.stack([(c + 1) % n, (c - 1) % n, (c + n // 2) % n], axis=1).astype(np.int64)


def _synthetic_grids(n_levels=3, n_base=N_BASE):
    grids = {}
    for level in range(n_levels):
        n = n_base * 4**level
        lon = (np.arange(n, dtype=np.float32) / n) * 2 * np.pi - np.pi
        grids[level] = LevelGrid(level=level, neighbors=_neighbors(n), lon=lon, lat=np.zeros(n, np.float32))
    return grids


def _write_grid(path, n, parent=None, coord_names=("lon_cell_centre", "lat_cell_centre")):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("cell", n)
        ds.createDimension("nv", 3)
        nbr = ds.createVariable("neighbor_cell_index", "i4", ("nv", "cell"))
        nbr[:] = (_neighbors(n) + 1).T
        for name in coord_names:
            ds.createVariable(name, "f8", ("cell",))[:] = np.linspace(-1, 1, n)
        if parent is not None:
            ds.createVariable("parent_cell_index", "i4", ("cell",))[:] = parent


# --------------------------------------------------------------------------
# Grid files
# --------------------------------------------------------------------------


def test_check_index_nesting_accepts_nested_parents():
    check_index_nesting(np.arange(48) // 4 + 1, n_parent_cells=12, level=5)


def test_check_index_nesting_rejects_unnested_parents_and_wrong_counts():
    parent = np.arange(48) // 4 + 1
    parent[[0, 4]] = parent[[4, 0]]
    with pytest.raises(ValueError, match="not index-nested"):
        check_index_nesting(parent, n_parent_cells=12, level=5)
    with pytest.raises(ValueError, match="expected 4 x 13"):
        check_index_nesting(np.arange(48) // 4 + 1, n_parent_cells=13, level=5)


def test_load_level_grids_reads_all_levels_and_checks_them(tmp_path):
    paths = {4: tmp_path / "b4.nc", 5: tmp_path / "b5.nc", 6: tmp_path / "b6.nc"}
    _write_grid(paths[4], 12)
    _write_grid(paths[5], 48, parent=np.arange(48) // 4 + 1, coord_names=("clon", "clat"))
    _write_grid(paths[6], 192, parent=np.arange(192) // 4 + 1)
    grids = load_level_grids({k: str(v) for k, v in paths.items()}, ncells_global=192)
    assert sorted(grids) == [4, 5, 6]
    assert grids[6].n_cells == 192 and grids[6].neighbors.shape == (192, 3)
    assert np.array_equal(grids[5].neighbors, _neighbors(48))  # 1-based in file, 0-based here
    assert grids[5].lon.dtype == np.float32

    with pytest.raises(ValueError, match="ICON runs on 768"):
        load_level_grids({k: str(v) for k, v in paths.items()}, ncells_global=768)
    with pytest.raises(ValueError, match="every level"):
        load_level_grids({4: str(paths[4]), 6: str(paths[6])})


def test_load_level_grids_rejects_unnested_file(tmp_path):
    parent = np.arange(48) // 4 + 1
    parent[[1, 47]] = parent[[47, 1]]
    _write_grid(tmp_path / "b4.nc", 12)
    _write_grid(tmp_path / "b5.nc", 48, parent=parent)
    with pytest.raises(ValueError, match="not index-nested"):
        load_level_grids({4: str(tmp_path / "b4.nc"), 5: str(tmp_path / "b5.nc")})


def test_check_cell_centers_accepts_matching_and_rejects_permuted_cells():
    grid = _synthetic_grids(n_levels=2)[1]
    glb = np.array([3, 10, 40])
    lon = grid.lon[glb].astype(np.float64) + 2 * np.pi  # same point, other longitude branch
    check_cell_centers(grid, glb, lon, grid.lat[glb])
    with pytest.raises(ValueError, match="2 of 3 cell centers differ"):
        check_cell_centers(grid, glb, grid.lon[[10, 3, 40]], grid.lat[glb])


# --------------------------------------------------------------------------
# Ownership
# --------------------------------------------------------------------------


def test_owner_is_rank_with_most_fine_cells_lowest_rank_on_ties():
    refinement = 4
    owned = [np.array([0, 1, 2, 4, 5]), np.array([3, 6, 7])]  # backbone 0: 3 vs 1, backbone 1: 2 vs 2
    counts = np.stack([backbone_counts(o, 2, refinement) for o in owned])
    assert counts.tolist() == [[3, 2], [1, 2]]
    assert assign_backbone_owners(counts, refinement).tolist() == [0, 0]


def test_owner_assignment_rejects_cells_missing_or_owned_twice():
    with pytest.raises(ValueError, match="backbone cell 1 has 3"):
        assign_backbone_owners(np.array([[4, 2], [0, 1]]), refinement=4)
    with pytest.raises(ValueError, match="backbone cell 0 has 5"):
        assign_backbone_owners(np.array([[4, 2], [1, 2]]), refinement=4)


# --------------------------------------------------------------------------
# Nested patch
# --------------------------------------------------------------------------


def test_patch_cell_ids_are_nested_descendant_blocks():
    ids = patch_cell_ids(np.array([2, 5]), k=2)
    assert ids.tolist() == list(range(32, 48)) + list(range(80, 96))
    assert np.array_equal(ids[16:32] // 16, np.full(16, 5))


def test_nested_mgrid_adjacency_matches_global_neighbors_inside_patch():
    grids = _synthetic_grids()
    backbone = np.array([7, 1, 2, 3])  # unsorted on purpose
    mg = build_nested_mgrid(grids, backbone)
    assert mg.backbone_ids.tolist() == [1, 2, 3, 7]
    assert (mg.base_level, mg.fine_level, mg.n_backbone, mg.refinement, mg.n_fine) == (0, 2, 4, 16, 64)

    for level in range(3):
        ids = patch_cell_ids(mg.backbone_ids, level)
        adjc, mask = mg.adjc[level].numpy(), mg.adjc_mask[level].numpy()
        assert adjc.shape == (ids.size, 9) and mask.shape == (ids.size, 9)
        assert np.array_equal(adjc[:, 0], np.arange(ids.size)) and not mask[:, 0].any()
        assert not mask[:, [2, 6]].any()
        global_nbr = grids[level].neighbors[ids][:, [0, 1, 2, 0, 1, 2, 0, 1]]
        inside = np.isin(global_nbr, ids)
        assert np.array_equal(ids[adjc[:, 1:]][inside], global_nbr[inside])
        assert np.array_equal(adjc[:, 1:][~inside], np.broadcast_to(np.arange(ids.size)[:, None], inside.shape)[~inside])
        for col in (1, 3, 4, 5, 7, 8):
            assert np.array_equal(mask[:, col], ~inside[:, col - 1])
        torch.testing.assert_close(mg.coords[level][:, 0], torch.as_tensor(grids[level].lon[ids]))


def test_fieldspace_mgrids_are_copies():
    mg = build_nested_mgrid(_synthetic_grids(), np.arange(9))
    mgrids = mg.fieldspace_mgrids()
    assert [m["zoom"] for m in mgrids] == [0, 1, 2]
    mgrids[1]["adjc"][:, 1] = -5
    assert (mg.adjc[1][:, 1] >= 0).all()


def test_patch_positions_and_cells_outside_patch():
    backbone = np.array([1, 4])
    assert patch_positions(np.array([16, 31, 64, 79]), backbone, refinement=16).tolist() == [0, 15, 16, 31]
    with pytest.raises(ValueError, match="1 fine cells are not in this rank's patch"):
        patch_positions(np.array([16, 32]), backbone, refinement=16)


# --------------------------------------------------------------------------
# Exchange
# --------------------------------------------------------------------------


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
    # "halo" rows that must never be sent.
    local = []
    for r in range(size):
        owned_glb = np.flatnonzero(rank_of_cell == r)
        halo_glb = rng.choice(np.flatnonzero(rank_of_cell != r), 5, replace=False)
        glb = rng.permutation(np.concatenate([owned_glb, halo_glb]))
        values = field[glb].copy()
        values[np.isin(glb, halo_glb)] = np.nan
        owned_local = np.flatnonzero(np.isin(glb, owned_glb))
        local.append((glb, values, owned_local))

    counts = np.stack([backbone_counts(glb[ol], N_BASE, refinement) for glb, _, ol in local])
    owner = assign_backbone_owners(counts, refinement)

    def rank_fn(comm):
        glb, values, owned_local = local[comm.Get_rank()]
        backbone = np.flatnonzero(owner == comm.Get_rank())
        ex = CellExchange.build(comm, owned_local, glb[owned_local], owner, backbone, refinement)
        patch = ex.to_patch(comm, values)
        back = ex.from_patch(comm, patch * 10.0)
        return backbone, patch, ex, back

    covered = []
    for r, (backbone, patch, ex, back) in enumerate(_run_ranks(size, rank_fn)):
        ids = patch_cell_ids(backbone, 2)
        assert np.array_equal(patch, field[ids])  # nested order, no halo values
        covered.append(ids)
        glb = local[r][0]
        assert np.array_equal(np.sort(ex.send_local_ids), local[r][2])
        assert np.array_equal(back, field[glb[ex.send_local_ids]] * 10.0)
    assert np.array_equal(np.sort(np.concatenate(covered)), np.arange(n_fine))


def test_exchange_build_fails_if_a_fine_cell_is_lost():
    size, refinement = 2, 4
    n_fine = N_BASE * refinement
    owner = np.array([0] * 6 + [1] * 6)

    def rank_fn(comm):
        owned = np.arange(0, n_fine, 2) if comm.Get_rank() == 0 else np.arange(1, n_fine, 2)
        if comm.Get_rank() == 1:
            owned = owned[:-1]  # one fine cell owned by nobody
        CellExchange.build(comm, np.arange(owned.size), owned, owner, np.flatnonzero(owner == comm.Get_rank()), refinement)

    with pytest.raises(ValueError, match="missing or duplicated"):
        _run_ranks(size, rank_fn)
