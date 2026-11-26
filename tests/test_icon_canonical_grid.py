"""Tests for FieldSpaceNN's icon_grid.py and the ICON path of its GridLayer:
the canonical cell order, the 12 neighbours and the ring of a token.

On a small icosahedral grid whose children and vertices are shuffled like an
ICON grid file's (tests/icosahedron.py). "The same place" is checked in each
cell's own frame: barycentric coordinates with respect to its corners A, B, C.
"""

import numpy as np
import pytest
import torch
import xarray as xr

from fieldspacenn.src.modules.field_space.field_space_base import Tokenizer
from fieldspacenn.src.modules.grids.grid_layer import GridLayer, tokens_with_ring
from fieldspacenn.src.modules.grids.grid_utils import healpix_grid_to_mgrid
from fieldspacenn.src.modules.grids.icon_grid import (
    canonical_mgrid,
    canonical_order,
    icon_canonical_mgrid,
    parent_corners,
    ring12,
)
from icosahedron import icosahedral_grid, is_counter_clockwise

N_LEVELS, N_ZOOMS = 4, 3  # 5120 cells; zoom 0 has 320


@pytest.fixture(scope="module")
def grid():
    return icosahedral_grid(N_LEVELS)


@pytest.fixture(scope="module")
def levels(grid):
    return canonical_order(*grid, N_ZOOMS)


def _centres(corners, vertex_xyz):
    centre = vertex_xyz[corners].sum(1)
    return centre / np.linalg.norm(centre, axis=-1, keepdims=True)


def _in_frame(frame_corners, vertex_xyz, points):
    """Barycentric coordinates (weights of B and C) of ``points``
    (n, k, 3) in the frames ``frame_corners`` (n, 3)."""
    a, b, c = (vertex_xyz[frame_corners[:, k]] for k in range(3))
    basis = np.stack([b - a, c - a], -1)  # (n, 3, 2)
    return np.einsum("nij,nkj->nki", np.linalg.pinv(basis), points - a[:, None])


def _same_places(places):
    """Every entry of every row of ``places`` (n, k, 2) is nearer to its own
    column's usual place than to any other column's."""
    usual = np.median(places, 0)
    nearest = np.linalg.norm(places[:, :, None] - usual[None, None], axis=-1).argmin(-1)
    return (nearest == np.arange(places.shape[1])).all()


def test_parent_corners_recovers_the_coarser_grid():
    coarse, xyz = icosahedral_grid(2, seed=1)
    fine, xyz = icosahedral_grid(3, seed=1)  # same vertices, plus the new midpoints
    parents, corner = parent_corners(fine)
    assert np.array_equal(np.sort(parents, 1), np.sort(coarse, 1))
    assert is_counter_clockwise(parents, xyz).all()
    assert (np.sort(corner, 1) == [-1, 0, 1, 2]).all()  # one centre child, one child per corner
    rows, kids = np.nonzero(corner >= 0)
    assert (fine.reshape(-1, 4, 3)[rows, kids] == parents[rows, corner[rows, kids]][:, None]).any(1).all()


def test_order_is_nested_and_a_permutation(grid, levels):
    for zoom, level in enumerate(levels):
        n, per_cell = len(level["file_index"]), 4**zoom
        assert np.array_equal(np.sort(level["file_index"]), np.arange(n))
        assert np.array_equal(level["file_index"] // per_cell, np.arange(n) // per_cell)  # stays in its zoom-0 cell
        assert is_counter_clockwise(level["corners"], grid[1]).all()
    # The parent of canonical cell c is canonical cell c // 4: its corners come from its children.
    for coarse, fine in zip(levels, levels[1:]):
        assert np.array_equal(np.sort(parent_corners(fine["corners"])[0], 1), np.sort(coarse["corners"], 1))


def test_every_zoom0_cell_has_its_cells_at_the_same_places(grid, levels):
    _, xyz = grid
    for zoom, level in enumerate(levels[1:], start=1):
        centres = _centres(level["corners"], xyz).reshape(-1, 4**zoom, 3)
        assert _same_places(_in_frame(levels[0]["corners"], xyz, centres))
        # In the file's order they are not.
        file_order = centres.reshape(-1, 3)[np.argsort(level["file_index"])].reshape(centres.shape)
        assert not _same_places(_in_frame(levels[0]["corners"], xyz, file_order))


def test_child_frames_follow_the_parent(grid, levels):
    """Child 0 is the centre child; child 1/2/3 holds the parent's A/B/C as
    its own A/B/C."""
    for coarse, fine in zip(levels, levels[1:]):
        children = fine["corners"].reshape(-1, 4, 3)
        assert not (children[:, 0, :, None] == coarse["corners"][:, None, :]).any()
        for k in range(3):
            assert np.array_equal(children[:, 1 + k, k], coarse["corners"][:, k])


def test_corner_A_lies_north_or_south(grid):
    mgrid = canonical_mgrid(*grid, N_ZOOMS)
    north, east = mgrid[0]["frame_direction"].T
    assert torch.allclose(north**2 + east**2, torch.ones_like(north), atol=1e-5)
    assert (east.abs() <= np.sin(np.radians(40))).all()  # 36 degrees at most, next to the poles
    assert (north > 0).sum() == (north < 0).sum()  # as many triangles point up as down


def test_ring12_has_the_touching_cells_in_the_same_order_for_every_cell(grid, levels):
    _, xyz = grid
    corners = levels[-1]["corners"]
    adjc, missing = ring12(corners)
    assert np.array_equal(adjc[:, 0], np.arange(len(adjc))) and not missing[:, 0].any()
    assert missing.sum() == 60 and (adjc[missing] == np.nonzero(missing)[0]).all()  # 5 cells at each of the 12 icosahedron corners
    shared = (corners[adjc][:, :, :, None] == corners[:, None, None, :]).any(-1).sum(-1)  # vertices shared with the cell
    assert (shared[:, [1, 5, 9]] == 2).all()  # across AB, BC, CA
    others = np.delete(np.arange(13), [0, 1, 5, 9])
    assert (shared[:, others][~missing[:, others]] == 1).all()
    complete = ~missing.any(1)
    assert all(len(set(row)) == 13 for row in adjc[complete])
    places = _in_frame(corners[complete], xyz, _centres(corners, xyz)[adjc[complete]])
    assert _same_places(places)


def test_tokens_with_ring_lists_own_cells_then_the_cells_around_the_token(grid, levels):
    _, xyz = grid
    for zoom, level in enumerate(levels):
        m = 4**zoom
        adjc, missing = (torch.from_numpy(a) for a in ring12(level["corners"]))
        table = tokens_with_ring(adjc, ~missing, m)
        own, ring = table[:, :m], table[:, m:]
        token = torch.arange(len(table))[:, None]
        assert torch.equal(own, torch.arange(len(adjc)).view(-1, m))
        assert ring.shape[1] == 6 * 2**zoom + 6
        outside = ring // m != token  # a missing ring cell is replaced by an own cell
        for t in range(len(table)):
            around = adjc[own[t]][~missing[own[t]]]
            around = set(around[around // m != t].tolist())
            assert set(ring[t][outside[t]].tolist()) == around and outside[t].sum() == len(around)
        complete = outside.all(1).numpy()
        assert complete.sum() == len(table) - 60
        centres = _centres(level["corners"], xyz)[table.numpy()]
        assert _same_places(_in_frame(levels[0]["corners"][complete], xyz, centres[complete]))


def test_grid_layer_gathers_icon_tokens_with_their_ring(grid):
    mgrid = canonical_mgrid(*grid, N_ZOOMS)
    fine = mgrid[-1]
    layer = GridLayer(2, fine["adjc"].clone(), fine["adjc_mask"].clone(), fine["coords"], grid_type="icon")
    x = torch.randn(2, len(fine["adjc"]), 3)
    out, mask = layer.get_global_with_nh(x, zoom_patch_out=0)
    table = tokens_with_ring(fine["adjc"], ~fine["adjc_mask"], 16)
    assert mask is None and torch.equal(out, x[:, table])
    assert layer.get_number_of_points_in_patch(0) == 16 + 30
    assert layer.get_number_of_points_in_patch(2) == 1 + 12
    with pytest.raises(NotImplementedError):
        layer.apply_shift(x, "north")


def test_tokenizer_appends_the_ring_at_every_zoom(grid):
    """FieldSpaceNN's token overlap on the ICON grid: a zoom-0 token of the
    field at any zoom is its own cells, then the cells around it."""
    mgrid = canonical_mgrid(*grid, N_ZOOMS)
    layers = {
        str(zoom): GridLayer(zoom, g["adjc"].clone(), g["adjc_mask"].clone(), g["coords"], grid_type="icon")
        for zoom, g in enumerate(mgrid)
    }
    sample_configs = {zoom: {"zoom_patch_sample": -1, "n_past_ts": 1, "n_future_ts": 0} for zoom in range(N_ZOOMS)}
    for zoom, g in enumerate(mgrid):
        tokenizer = Tokenizer([zoom], 0, overlap_thickness=1, grid_layers=layers)
        x = torch.randn(1, 1, 1, len(g["adjc"]), 1, 1)  # (b, v, t, n, d, f)
        tokens = tokenizer({zoom: x}, sample_configs)  # (b, v, T, N, D, t, n, d, f)
        table = tokens_with_ring(g["adjc"], ~g["adjc_mask"], 4**zoom)
        assert tokenizer.token_size[1] == 4**zoom + 6 * 2**zoom + 6
        assert torch.equal(tokens[0, 0, 0, :, 0, 0, :, 0, 0], x[0, 0, 0, :, 0, 0][table])


def _healpix_tokens_before(layer, x, zoom_patch_out):
    """`GridLayer.get_global_with_nh` as it was before the ICON path."""
    indices = layer.adjc if zoom_patch_out == layer.zoom else layer.adjc[:, [2, 4, 6, 8]]
    size = 4 ** (layer.zoom - zoom_patch_out)
    starts = torch.arange(0, indices.shape[0] + size, size).view(-1, 1, 1)
    indices = indices.view(-1, size, indices.shape[-1])
    out_of_patch = (indices < starts[:-1]) | (indices >= starts[1:])
    nh = indices[out_of_patch].view(indices.shape[0], int(out_of_patch.sum(dim=[-1, -2])[0]))
    return x[:, torch.concat((indices[:, :, 0].view(indices.shape[0], -1), nh), dim=-1)]


def test_healpix_tokens_are_unchanged():
    healpix = healpix_grid_to_mgrid(2)[2]
    layer = GridLayer(2, healpix["adjc"], healpix["adjc_mask"], healpix["coords"])
    x = torch.randn(1, len(healpix["adjc"]), 2)
    for zoom_patch_out in (1, 2):
        assert torch.equal(layer.get_global_with_nh(x, zoom_patch_out=zoom_patch_out)[0], _healpix_tokens_before(layer, x, zoom_patch_out))


def test_grid_file_is_read_in_canonical_order(grid, tmp_path):
    vertex_of_cell, xyz = grid
    centre = _centres(vertex_of_cell, xyz)
    path = tmp_path / "grid.nc"
    xr.Dataset(
        {
            "vertex_of_cell": (("nv", "cell"), vertex_of_cell.T + 1),
            **{f"cartesian_{axis}_vertices": ("vertex", xyz[:, k]) for k, axis in enumerate("xyz")},
            "lon_cell_centre": ("cell", np.arctan2(centre[:, 1], centre[:, 0])),
            "lat_cell_centre": ("cell", np.arcsin(centre[:, 2])),
        }
    ).to_netcdf(path)
    from_file, from_arrays = icon_canonical_mgrid(str(path), N_ZOOMS), canonical_mgrid(*grid, N_ZOOMS)
    for a, b in zip(from_file, from_arrays):
        assert a["grid_type"] == "icon" and torch.equal(a["adjc"], b["adjc"]) and torch.equal(a["file_index"], b["file_index"])
        assert torch.allclose(a["coords"], b["coords"], atol=1e-5)


@pytest.mark.parametrize("scope", ["local", "global"])
def test_local_attention_links_a_token_to_its_12_neighbours_only(scope):
    """One reconstructor-style attention block (tokens on zoom 0, fields on zooms 0 and 1) with
    random weights, so its gates are open: a change in one token's cells reaches, with local
    attention, exactly that token and its 12 neighbours; with global attention, every token."""
    from messe.reconstructor_forecaster.model import _BlockConfigs, _sample_configs
    from fieldspacenn.src.models.mg_transformer.mg_base_model import MG_base_model, create_encoder_decoder_block

    torch.manual_seed(0)
    mgrid = canonical_mgrid(*icosahedral_grid(3), 2)  # zoom 0: 320 tokens, zoom 1: 1280 cells
    layers = MG_base_model(mgrid).grid_layers
    configs = _BlockConfigs(16, 4, 16, 8, 0, attention=scope)
    block = create_encoder_decoder_block(configs.attention([0, 1]), [0, 1], [1, 1], [1], grid_layers=layers, n_head_channels=4)
    with torch.no_grad():
        for p in block.parameters():
            p.normal_(0, 0.3)

    def run(x):
        emb = {"TimeEmbedder": {z: torch.zeros(1, 1) for z in (0, 1)}}
        with torch.no_grad():
            return block([{z: t.clone() for z, t in x.items()}], sample_configs=_sample_configs(1, 0, [0, 1]), emb_groups=[emb])[0]

    x = {z: torch.randn(1, 1, 1, 320 * 4**z, 1, 1) for z in (0, 1)}
    token = int(torch.nonzero(~mgrid[0]["adjc_mask"].any(1))[0])  # a token with all 12 neighbours
    changed = {z: t.clone() for z, t in x.items()}
    for z, t in changed.items():  # random: a constant shift is removed by the layer norm before q, k, v
        t[0, 0, 0, token * 4**z: (token + 1) * 4**z] += torch.randn(4**z, 1, 1)
    a, b = run(x), run(changed)
    reached = set(torch.nonzero((a[0] - b[0]).abs().view(-1) > 1e-6).view(-1).tolist())
    window = set(mgrid[0]["adjc"][token].tolist())  # the token and its 12 neighbours
    assert len(window) == 13
    if scope == "local":
        assert reached == window
    else:
        assert len(reached) > 200


@pytest.mark.parametrize("token_len_time", [3, 6])
def test_local_forecaster_links_old_time_tokens_to_the_forecast_in_the_window_only(token_len_time):
    """The forecaster with local attention, 5 context slots + the future slot, random weights (open
    gates): a change in the oldest context slot of one cell reaches the forecast of exactly that
    cell and its 12 neighbours. With token_len_time 3 the oldest slot is in the other time token
    than the forecast, so it gets there only by attention over time tokens; with 6 (packed) it is
    in the forecast's own token."""
    from messe.reconstructor_forecaster.model import Forecaster, _BlockConfigs
    from fieldspacenn.src.models.mg_transformer.mg_base_model import MG_base_model

    torch.manual_seed(0)
    mgrid = canonical_mgrid(*icosahedral_grid(3), 2)  # zoom 0: 320 tokens, zoom 1: 1280 cells
    layers = MG_base_model(mgrid).grid_layers
    n_context = 5
    model = Forecaster(layers, {0: 1, 1: 1}, n_context, 1, _BlockConfigs(16, 4, 16, 8, 0, attention="local"), token_len_time)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)

    def run(context):
        emb = {"TimeEmbedder": {z: torch.arange(n_context + 1.0).view(1, -1) / 144 for z in (0, 1)}}
        with torch.no_grad():
            return model({z: t.clone() for z, t in context.items()}, emb)

    context = {z: torch.randn(1, 1, n_context, 320 * 4**z, 1, 1) for z in (0, 1)}
    token = int(torch.nonzero(~mgrid[0]["adjc_mask"].any(1))[0])  # a token with all 12 neighbours
    changed = {z: t.clone() for z, t in context.items()}
    for z, t in changed.items():  # only the oldest slot
        t[0, 0, 0, token * 4**z: (token + 1) * 4**z] += torch.randn(4**z, 1, 1)
    a, b = run(context), run(changed)
    assert a[0].shape == (1, 1, 1, 320, 1, 1)  # the future slot only
    reached = set(torch.nonzero((a[0] - b[0]).abs().view(-1) > 1e-6).view(-1).tolist())
    assert reached == set(mgrid[0]["adjc"][token].tolist())


def test_forecaster_token_len_time_must_divide_the_slots():
    from messe.reconstructor_forecaster.model import Forecaster, _BlockConfigs
    from fieldspacenn.src.models.mg_transformer.mg_base_model import MG_base_model

    layers = MG_base_model(canonical_mgrid(*icosahedral_grid(2), 2)).grid_layers
    with pytest.raises(ValueError, match="must divide n_context \\+ 1 = 6"):
        Forecaster(layers, {0: 1, 1: 1}, 5, 1, _BlockConfigs(16, 4, 16, 8, 0), token_len_time=4)
