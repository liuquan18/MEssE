"""Unit tests for icon_mgrid_utils.build_local_mgrid, using a synthetic
stand-in for COMIN's `domain.cells` instead of a live ICON simulation.

Scenario: nproma=11, one block (nblks=1) -> 11 local node slots, all valid
(no padding). Local flat id == array position (blk=1 makes flat_index(p+1,
1, 11) == p). Three global coarse parents (0, 1, 2); global fine cells
0-3 belong to parent 0, 4-7 to parent 1, 8-11 to parent 2.

This rank's local (owned+halo) fine cells are global fine 0-10 (local
positions 0-10, in the same order) -- i.e. every child of parents 0 and 1
is present locally (two COMPLETE groups), but only 3 of parent 2's 4
children are present (global fine 11 was never shipped to this rank -- the
expected signature of a partition boundary cutting through a quad, not a
bug). So build_local_mgrid should keep exactly 2 coarse tokens (parents 0
and 1) and drop parent 2's 3 orphan fine cells entirely.

Neighbor links: a self-contained cycle within each of parent 0's and
parent 1's groups (intra-group, exercises normal remapping), one
CROSS-group edge from local cell 3 (last child of parent 0) to local cell 4
(first child of parent 1) (exercises fine_adjc correctly remapping a
same-rank, different-token neighbor rather than masking it), a cycle within
parent 2's partial group with one dangling "neighbor not shipped" slot
(exercises the missing-neighbor fallback), and no other links.
"""
#%%
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icon_mgrid_utils import (
    CoarseGrid,
    build_fine_adjacency,
    build_local_mgrid,
    check_grid_files,
    pool_fine_to_coarse,
)

#%%
def _make_domain():
    nproma, nblks, n_nbr = 11, 1, 3

    neighbor_idx = np.zeros((nproma, nblks, n_nbr), dtype=np.int64)
    neighbor_blk = np.zeros((nproma, nblks, n_nbr), dtype=np.int64)

    def link(p_src, p_dst, slot):
        neighbor_idx[p_src, 0, slot] = p_dst + 1  # 1-based idx
        neighbor_blk[p_src, 0, slot] = 1

    # parent-0 group (local 0,1,2,3): cycle on slot 0.
    link(0, 1, 0)
    link(1, 2, 0)
    link(2, 3, 0)
    link(3, 0, 0)
    # cross-group edge: local 3 -> local 4 (parent 0's last child -> parent
    # 1's first child), on slot 1.
    link(3, 4, 1)
    # parent-1 group (local 4,5,6,7): cycle on slot 0.
    link(4, 5, 0)
    link(5, 6, 0)
    link(6, 7, 0)
    link(7, 4, 0)
    # parent-2 partial group (local 8,9,10; global fine 11 never shipped):
    # cycle on slot 0, with local 10's "4th" neighbor left as (0,0) --
    # modeling the missing, not-locally-present 4th sibling.
    link(8, 9, 0)
    link(9, 10, 0)
    # local 10's slot 0 left at (0,0): no neighbor shipped for the missing sibling.

    clon = np.arange(11, dtype=np.float64) * 0.1
    clat = np.zeros(11, dtype=np.float64)
    # decomp_domain: 0 = owned, >0 = halo. All 11 cells are valid (owned+halo).
    decomp_domain = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    glb_index = np.arange(1, 12, dtype=np.int64)  # 1-based, local i -> global fine i

    cells = SimpleNamespace(
        ncells=11,
        nblks=nblks,
        clon=clon,
        clat=clat,
        decomp_domain=decomp_domain,
        neighbor_idx=neighbor_idx,
        neighbor_blk=neighbor_blk,
        glb_index=glb_index,
    )
    return SimpleNamespace(cells=cells)


def _make_coarse_grid():
    # 3 global coarse cells: 0 <-> 1 (real neighbors of each other via col 1),
    # and both also (falsely, for test purposes) list 2 as a neighbor via
    # col 2, to exercise the "neighbor exists globally but isn't a locally
    # kept token" fallback. All other columns masked/invalid.
    adjc = np.zeros((3, 9), dtype=np.int64)
    adjc_mask = np.ones((3, 9), dtype=bool)
    adjc_mask[:, 0] = False  # self is always valid

    adjc[0] = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    adjc_mask[0, 1] = False
    adjc_mask[0, 2] = False

    adjc[1] = [1, 0, 2, 0, 0, 0, 0, 0, 0]
    adjc_mask[1, 1] = False
    adjc_mask[1, 2] = False

    adjc[2] = [2, 0, 1, 0, 0, 0, 0, 0, 0]
    adjc_mask[2, 1] = False
    adjc_mask[2, 2] = False

    return CoarseGrid(n_cells=3, adjc=adjc, adjc_mask=adjc_mask)


def _make_parent_index_global():
    # 0-based global coarse parent per 0-based global fine cell, for 12
    # global fine cells (0-11): fine 0-3 -> parent 0, 4-7 -> parent 1,
    # 8-11 -> parent 2.
    return np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)

#%%
def test_only_complete_groups_are_kept():
    domain = _make_domain()
    mgrid = build_local_mgrid(
        domain, nproma=11, coarse_grid=_make_coarse_grid(),
        parent_index_global=_make_parent_index_global(), device=torch.device("cpu"),
    )

    assert mgrid.n_coarse == 2
    assert mgrid.n_fine_kept == 8
    assert mgrid.parent_group_size_histogram == {4: 2, 3: 1}

    # Every kept fine local id belongs to parent 0 or parent 1 (local ids 0-7);
    # none of parent 2's orphan cells (local ids 8, 9, 10) are kept.
    kept = set(mgrid.fine_local_ids_kept.tolist())
    assert kept == set(range(8))


def test_fine_local_ids_kept_groups_of_four_match_their_parent():
    domain = _make_domain()
    mgrid = build_local_mgrid(
        domain, nproma=11, coarse_grid=_make_coarse_grid(),
        parent_index_global=_make_parent_index_global(), device=torch.device("cpu"),
    )
    ids = mgrid.fine_local_ids_kept.tolist()
    group0, group1 = frozenset(ids[0:4]), frozenset(ids[4:8])
    assert {group0, group1} == {frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7})}


def test_fine_adjc_intra_group_and_cross_group_remap():
    domain = _make_domain()
    mgrid = build_local_mgrid(
        domain, nproma=11, coarse_grid=_make_coarse_grid(),
        parent_index_global=_make_parent_index_global(), device=torch.device("cpu"),
    )
    ids = mgrid.fine_local_ids_kept.tolist()
    full_to_compact = {full_id: compact for compact, full_id in enumerate(ids)}

    # local cell 3's slot-1 cross-group neighbor (local cell 4) must resolve
    # to compact id of local cell 4, not fall back to self/mask.
    c3 = full_to_compact[3]
    c4 = full_to_compact[4]
    assert mgrid.fine_adjc[c3, 2].item() == c4  # col index 2 == tile slot 1 (col 0 is self)
    assert not mgrid.fine_adjc_mask[c3, 2].item()

    # local cell 3's slot-0 intra-group neighbor (local cell 0, closing the cycle).
    c0 = full_to_compact[0]
    assert mgrid.fine_adjc[c3, 1].item() == c0
    assert not mgrid.fine_adjc_mask[c3, 1].item()


def test_coarse_adjc_remaps_kept_neighbor_and_masks_unkept_one():
    domain = _make_domain()
    mgrid = build_local_mgrid(
        domain, nproma=11, coarse_grid=_make_coarse_grid(),
        parent_index_global=_make_parent_index_global(), device=torch.device("cpu"),
    )
    # kept_parents are sorted ascending -> token 0 = global parent 0, token 1 = global parent 1.
    # token 0's real neighbor (global parent 1) is kept -> should remap to token 1.
    assert mgrid.coarse_adjc[0, 1].item() == 1
    assert not mgrid.coarse_adjc_mask[0, 1].item()
    # token 0's other "neighbor" (global parent 2) is NOT kept -> self + masked.
    assert mgrid.coarse_adjc[0, 2].item() == 0
    assert mgrid.coarse_adjc_mask[0, 2].item()
    # token 1 symmetric: real neighbor is token 0.
    assert mgrid.coarse_adjc[1, 1].item() == 0
    assert not mgrid.coarse_adjc_mask[1, 1].item()


def test_coarse_coords_are_mean_of_children():
    domain = _make_domain()
    mgrid = build_local_mgrid(
        domain, nproma=11, coarse_grid=_make_coarse_grid(),
        parent_index_global=_make_parent_index_global(), device=torch.device("cpu"),
    )
    ids = mgrid.fine_local_ids_kept.tolist()
    expected_lon = {
        0: sum(0.1 * i for i in ids[0:4]) / 4.0,
        1: sum(0.1 * i for i in ids[4:8]) / 4.0,
    }
    for t in (0, 1):
        assert mgrid.coarse_coords[t, 0].item() == pytest.approx(expected_lon[t], abs=1e-6)
        assert mgrid.coarse_coords[t, 1].item() == pytest.approx(0.0, abs=1e-6)


def test_pool_fine_to_coarse_means_each_group_of_four():
    domain = _make_domain()
    mgrid = build_local_mgrid(
        domain, nproma=11, coarse_grid=_make_coarse_grid(),
        parent_index_global=_make_parent_index_global(), device=torch.device("cpu"),
    )
    ids = mgrid.fine_local_ids_kept.tolist()
    fine_x_full = torch.arange(11, dtype=torch.float32) * 10.0  # value = local_id * 10
    fine_x_compact = fine_x_full[torch.as_tensor(ids)].unsqueeze(-1)  # (8, 1)

    pooled = pool_fine_to_coarse(fine_x_compact, mgrid.n_coarse)
    assert pooled.shape == (2, 1)

    expected0 = sum(ids[0:4]) * 10.0 / 4.0
    expected1 = sum(ids[4:8]) * 10.0 / 4.0
    assert pooled[0, 0].item() == pytest.approx(expected0, abs=1e-4)
    assert pooled[1, 0].item() == pytest.approx(expected1, abs=1e-4)


def test_fine_compact_owned_mask_matches_decomp_domain():
    domain = _make_domain()
    mgrid = build_local_mgrid(
        domain, nproma=11, coarse_grid=_make_coarse_grid(),
        parent_index_global=_make_parent_index_global(), device=torch.device("cpu"),
    )
    ids = mgrid.fine_local_ids_kept.tolist()
    # decomp_domain: local 0-5 owned, 6-10 halo.
    expected = [i < 6 for i in ids]
    assert mgrid.fine_compact_owned_mask.tolist() == expected


def test_fine_adjc_mask_never_masks_gridlayer_polar_fix_columns():
    """Regression test for job 27394205: FieldSpaceNN's GridLayer runs a
    fragile HEALPix "polar discontinuity" fixup (propagate_assignments) at
    adjacency columns 2 and 6 for any zoom at list-position >= 1 (always
    our fine zoom) -- it crashed on this exact synthetic domain's real
    missing-neighbor cases (cells 0-2 and 4-7 all have a genuinely missing
    slot-1/slot-2 neighbor, i.e. would be masked at columns 2 and/or 6
    without the fix in build_local_mgrid). Confirmed on a live run that
    this is not just a toy-test artifact. See
    icon_mgrid_utils._GRIDLAYER_POLAR_FIX_COLUMNS.
    """
    domain = _make_domain()
    mgrid = build_local_mgrid(
        domain, nproma=11, coarse_grid=_make_coarse_grid(),
        parent_index_global=_make_parent_index_global(), device=torch.device("cpu"),
    )
    assert not mgrid.fine_adjc_mask[:, 2].any()
    assert not mgrid.fine_adjc_mask[:, 6].any()
    # Sanity: this scenario really does have missing neighbors elsewhere
    # (columns 3, 5, 8 are untouched by the fix and should still show some
    # masking), so the assertions above are not vacuously true.
    assert mgrid.fine_adjc_mask[:, [1, 3, 4, 5, 7, 8]].any()


def test_build_fine_adjacency_multi_block_uses_correct_cell_ordering():
    """Regression test (Codex review finding, 2026-09-14): domain.cells.*
    arrays are (nproma, nblks) with idx (nproma) the fast-varying axis --
    matching flat_index's own convention, c = (blk-1)*nproma + (idx-1).
    Flattening/reshaping them with numpy's default order="C" instead
    varies blk fastest, silently permuting which physical cell ends up at
    flat position c whenever nblks > 1. Every synthetic domain elsewhere in
    this file uses nblks=1 (matching every real GPU run so far, which
    always sets nblocks_c=1), under which "C" and "F" order coincide -- so
    this is the one test here that actually exercises nblks > 1 and would
    have caught the bug (self-column values and clon would both have come
    out permuted under the old order="C" reshapes).
    """
    nproma, nblks, n_nbr = 4, 2, 3
    n_nodes = nproma * nblks

    # clon[idx-1, blk-1] = (idx-1) + (blk-1)*10 -- distinguishable per-cell
    # value so a wrong flattening order is easy to detect positionally.
    clon = np.zeros((nproma, nblks))
    for p in range(nproma):
        for b in range(nblks):
            clon[p, b] = p + b * 10
    clat = np.zeros((nproma, nblks))
    decomp_domain = np.zeros((nproma, nblks), dtype=np.int64)  # all owned

    neighbor_idx = np.zeros((nproma, nblks, n_nbr), dtype=np.int64)
    neighbor_blk = np.zeros((nproma, nblks, n_nbr), dtype=np.int64)
    # One real cross-block link: flat cell 0 (idx=1,blk=1) -> flat cell 5
    # (idx=2,blk=2), via slot 0. Everything else stays (0,0)/invalid.
    neighbor_idx[0, 0, 0] = 2
    neighbor_blk[0, 0, 0] = 2

    cells = SimpleNamespace(
        ncells=n_nodes,
        nblks=nblks,
        clon=clon,
        clat=clat,
        decomp_domain=decomp_domain,
        neighbor_idx=neighbor_idx,
        neighbor_blk=neighbor_blk,
    )
    domain = SimpleNamespace(cells=cells)

    fine = build_fine_adjacency(domain, nproma=nproma)

    # Positional check: flat cell c must have clon (c % nproma) + (c // nproma) * 10.
    expected_clon = [p + b * 10 for b in range(nblks) for p in range(nproma)]
    assert fine.clon.tolist() == pytest.approx(expected_clon)

    # Self-column check: adjc[c, 0] must be c itself, for every c.
    assert fine.adjc[:, 0].tolist() == list(range(n_nodes))

    # Cross-block neighbor check: flat cell 0's slot-0 neighbor (tile
    # column 1) must resolve to flat cell 5, unmasked.
    assert fine.adjc[0, 1] == 5
    assert not fine.adjc_mask[0, 1]


def test_check_grid_files_accepts_matching_pair():
    check_grid_files(_make_parent_index_global(), _make_coarse_grid(), ncells_global=12)


def test_check_grid_files_rejects_fine_grid_of_another_resolution():
    # e.g. R2B4 grid files on an R2B8 run
    with pytest.raises(ValueError, match="ICON runs on 48"):
        check_grid_files(_make_parent_index_global(), _make_coarse_grid(), ncells_global=48)


def test_check_grid_files_rejects_coarse_grid_not_one_level_up():
    coarse = _make_coarse_grid()
    coarse_two_levels_up = CoarseGrid(n_cells=12, adjc=coarse.adjc, adjc_mask=coarse.adjc_mask)
    with pytest.raises(ValueError, match="expected 3"):
        check_grid_files(_make_parent_index_global(), coarse_two_levels_up, ncells_global=12)


def test_check_grid_files_rejects_parent_index_outside_coarse_grid():
    parent = _make_parent_index_global()
    parent[-1] = 3
    with pytest.raises(ValueError, match="outside the coarse grid"):
        check_grid_files(parent, _make_coarse_grid(), ncells_global=12)
