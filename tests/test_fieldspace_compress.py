"""Unit tests for :mod:`fieldspace_compress` -- staged Field-Space
compression/decompression over a multi-level ICON pyramid, plus the latent
reservoir.

Pure logic: no ``comin``, no MPI, no GPU, no live ICON simulation, and no
``FieldSpaceNN`` import. This is the fast feedback loop for this layer.
"""

import pytest
import torch

from fieldspace_compress import (
    CompressionStage,
    DecompressionStage,
    FieldSpaceAutoencoder,
    LatentReservoir,
    PyramidSpec,
    ancestor_broadcast_init_,
    assemble_patch,
    children_between,
    decompose_pyramid,
    disassemble_patch,
    enforce_conservation,
    recompose_pyramid,
    zero_fill_baseline,
)

# The R2B4/R2B3 prototype and the R2B8 target config.
PROTO = PyramidSpec((3, 4))
TARGET = PyramidSpec((3, 6, 7, 8))

# Real global ICON cell counts.
R2B3, R2B6, R2B7, R2B8 = 5_120, 327_680, 1_310_720, 5_242_880

NLEV = 2


def _x(spec, n_level0=2, nlev=NLEV, dtype=torch.float64):
    """A fine field whose cell count is consistent with ``spec``."""
    torch.manual_seed(0)
    n_fine = n_level0 * children_between(spec.zooms[0], spec.zooms[-1])
    return torch.randn(n_fine, nlev, dtype=dtype)


# --------------------------------------------------------------------------
# Pyramid bookkeeping
# --------------------------------------------------------------------------


def test_children_between_handles_multi_level_gaps():
    assert children_between(7, 8) == 4
    assert children_between(6, 8) == 16
    assert children_between(3, 6) == 64  # the target config's big jump
    assert children_between(5, 5) == 1


def test_group_sizes_of_the_target_pyramid():
    assert [TARGET.group_size(i) for i in (1, 2, 3)] == [64, 4, 4]
    assert PROTO.group_size(1) == 4


def test_pyramid_spec_rejects_bad_ladders():
    with pytest.raises(ValueError):
        PyramidSpec((5,))
    with pytest.raises(ValueError):
        PyramidSpec((6, 3))
    with pytest.raises(ValueError):
        PyramidSpec((4, 4))


def test_cells_at_reproduces_real_icon_counts():
    assert [TARGET.cells_at(i, R2B8) for i in range(4)] == [R2B3, R2B6, R2B7, R2B8]


# --------------------------------------------------------------------------
# The design sketch's data-size table (.claude/compression.png)
# --------------------------------------------------------------------------


def _stack_size(channels):
    return sum(TARGET.cells_at(i, R2B8) * c for i, c in enumerate(channels))


def test_decomposed_pyramid_is_1_3x_the_native_field():
    assert _stack_size([1, 1, 1, 1]) == 6_886_400
    assert _stack_size([1, 1, 1, 1]) / R2B8 == pytest.approx(1.31, abs=0.01)


def test_stage_sizes_match_the_design_sketch():
    # After stage 1: {x3, r6, r7' x4} -- level 8 dropped.
    assert _stack_size([1, 1, 4]) == 5_575_680
    assert _stack_size([1, 1, 4]) / R2B8 == pytest.approx(1.06, abs=0.01)
    # After stage 2: {x3, r6' x4} -- level 7 dropped.
    assert _stack_size([1, 4]) == 1_315_840
    assert _stack_size([1, 4]) / R2B8 == pytest.approx(0.25, abs=0.01)


def test_target_pyramid_is_critically_sampled():
    """Storage is 1.31x redundant; information is not redundant at all. Each
    residual level is zero-mean per parent group, so carries (g-1)/g free DOF."""
    dof = TARGET.cells_at(0, R2B8)
    for level in range(1, TARGET.n_levels):
        g = TARGET.group_size(level)
        dof += TARGET.cells_at(level, R2B8) * (g - 1) // g
    assert dof == R2B8


# --------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [PROTO, TARGET])
def test_decompose_recompose_is_identity(spec):
    x = _x(spec)
    levels = decompose_pyramid(x, spec)
    assert len(levels) == spec.n_levels
    torch.testing.assert_close(recompose_pyramid(levels, spec), x)


@pytest.mark.parametrize("spec", [PROTO, TARGET])
def test_every_residual_level_is_zero_mean_within_its_parent_group(spec):
    levels = decompose_pyramid(_x(spec), spec)
    for level in range(1, spec.n_levels):
        g = spec.group_size(level)
        r = levels[level]
        per_group = r.view(r.shape[0] // g, g, r.shape[1], r.shape[2]).mean(dim=1)
        torch.testing.assert_close(per_group, torch.zeros_like(per_group))


@pytest.mark.parametrize("spec", [PROTO, TARGET])
def test_decomposed_level_shapes(spec):
    x = _x(spec)
    levels = decompose_pyramid(x, spec)
    for i, lv in enumerate(levels):
        assert lv.shape == (spec.cells_at(i, x.shape[0]), 1, NLEV)


def test_recompose_rejects_multi_channel_levels():
    levels = decompose_pyramid(_x(PROTO), PROTO)
    levels[0] = levels[0].repeat(1, 3, 1)
    with pytest.raises(ValueError):
        recompose_pyramid(levels, PROTO)


# --------------------------------------------------------------------------
# Conservation projection
# --------------------------------------------------------------------------


def test_enforce_conservation_projects_and_is_idempotent():
    torch.manual_seed(1)
    levels = [
        torch.randn(TARGET.cells_at(i, 1024), 1, NLEV, dtype=torch.float64)
        for i in range(TARGET.n_levels)
    ]
    once = enforce_conservation(levels, TARGET)
    for level in range(1, TARGET.n_levels):
        g = TARGET.group_size(level)
        r = once[level]
        per_group = r.view(r.shape[0] // g, g, r.shape[1], r.shape[2]).mean(dim=1)
        torch.testing.assert_close(per_group, torch.zeros_like(per_group))
    for a, b in zip(enforce_conservation(once, TARGET), once):
        torch.testing.assert_close(a, b)


def test_enforce_conservation_leaves_coarse_mean_untouched():
    levels = decompose_pyramid(_x(TARGET), TARGET)
    torch.testing.assert_close(enforce_conservation(levels, TARGET)[0], levels[0])


# --------------------------------------------------------------------------
# Patch assembly (eqs 4-5, 8, 11)
# --------------------------------------------------------------------------


def test_patch_width_matches_paper_and_sketch():
    # Single-channel inputs, g=4: the paper's own 1 + 4.
    assert PROTO.patch_in_channels(1, c_target=1, c_drop=1) == 5
    # Sketch stage 2 consumes a 4-channel R2B7': 1 + 4*4 = 17.
    assert TARGET.patch_in_channels(2, c_target=1, c_drop=4) == 17
    # The target pyramid's big 3 -> 6 jump: 1 + 64 children in one fold.
    assert TARGET.patch_in_channels(1, c_target=1, c_drop=1) == 65
    # The paper's own three-level example done in one fold: 1 + 4 + 16 = 21
    # is the same width as folding z=8 into z=6 directly (1 + 16).
    assert PyramidSpec((6, 8)).patch_in_channels(1, c_target=1, c_drop=1) == 17


def test_assemble_disassemble_round_trip():
    target = torch.randn(5, 3, NLEV, dtype=torch.float64)
    drop = torch.randn(20, 2, NLEV, dtype=torch.float64)
    patch = assemble_patch(target, drop, group=4)
    assert patch.shape == (5, 3 + 4 * 2, NLEV)
    back_t, back_d = disassemble_patch(patch, c_target=3, c_drop=2, group=4)
    torch.testing.assert_close(back_t, target)
    torch.testing.assert_close(back_d, drop)


def test_patch_leading_channels_are_the_target_level():
    target = torch.randn(5, 3, NLEV, dtype=torch.float64)
    drop = torch.randn(20, 2, NLEV, dtype=torch.float64)
    torch.testing.assert_close(assemble_patch(target, drop, 4)[:, :3, :], target)


# --------------------------------------------------------------------------
# The staged autoencoder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_stages", [1, 2, 3])
def test_autoencoder_round_trips_shape_for_every_stage_count(n_stages):
    x = _x(TARGET, dtype=torch.float32)
    ae = FieldSpaceAutoencoder(TARGET, n_stages=n_stages, c_latent=4)
    assert ae(x).shape == x.shape
    assert len(ae.encode(x)) == TARGET.n_levels - n_stages


def test_autoencoder_rejects_impossible_stage_counts():
    with pytest.raises(ValueError):
        FieldSpaceAutoencoder(TARGET, n_stages=0)
    with pytest.raises(ValueError):
        FieldSpaceAutoencoder(TARGET, n_stages=TARGET.n_levels)


def test_two_stage_latent_matches_the_sketch_channel_layout():
    """Sketch endpoint: {x3 (1 ch), r6' (4 ch)}."""
    x = _x(TARGET, dtype=torch.float32)
    levels = FieldSpaceAutoencoder(TARGET, n_stages=2, c_latent=4).encode(x)
    assert [lv.shape[1] for lv in levels] == [1, 4]
    assert levels[0].shape[0] == TARGET.cells_at(0, x.shape[0])
    assert levels[1].shape[0] == TARGET.cells_at(1, x.shape[0])


@pytest.mark.parametrize("c_latent,expected_ratio", [(1, 4.0), (2, 2.0), (4, 1.0)])
def test_prototype_compression_ratio(c_latent, expected_ratio):
    """The R2B4/R2B3 knob table from the plan."""
    x = _x(PROTO, n_level0=8, dtype=torch.float32)
    ae = FieldSpaceAutoencoder(PROTO, n_stages=1, c_latent=c_latent)
    assert x.numel() / ae.latent_numel(x) == pytest.approx(expected_ratio)


def test_autoencoder_output_is_always_scale_conservative():
    x = _x(TARGET)
    ae = FieldSpaceAutoencoder(TARGET, n_stages=2, c_latent=4).double()
    levels = decompose_pyramid(ae(x), TARGET)
    for level in range(1, TARGET.n_levels):
        g = TARGET.group_size(level)
        r = levels[level]
        per_group = r.view(r.shape[0] // g, g, r.shape[1], r.shape[2]).mean(dim=1)
        torch.testing.assert_close(per_group, torch.zeros_like(per_group))


# --------------------------------------------------------------------------
# The learned stack strictly generalises zero-filling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec,n_stages", [(PROTO, 1), (TARGET, 1), (TARGET, 2)])
def test_zero_fill_init_reproduces_truncation_exactly(spec, n_stages):
    x = _x(spec)
    ae = FieldSpaceAutoencoder(spec, n_stages=n_stages, c_latent=4).double()
    ae.init_as_zero_fill()
    torch.testing.assert_close(ae(x), zero_fill_baseline(x, spec, n_stages))


def test_zero_fill_baseline_discards_exactly_the_dropped_levels():
    x = _x(TARGET)
    levels = decompose_pyramid(zero_fill_baseline(x, TARGET, n_stages=2), TARGET)
    kept = decompose_pyramid(x, TARGET)
    torch.testing.assert_close(levels[0], kept[0])
    torch.testing.assert_close(levels[1], kept[1])
    for level in (2, 3):
        torch.testing.assert_close(levels[level], torch.zeros_like(levels[level]))


def test_ancestor_broadcast_init_needs_room_for_the_target_level():
    """A narrower latent than the target level cannot pass it through, so the
    zero-fill warm start is not expressible and the helper must say so.

    Note this is unreachable through FieldSpaceAutoencoder: stages always target
    a level that still carries its original single channel (it is the *dropped*
    level that arrives widened), so c_target == 1 <= c_latent always. The guard
    exists for stages built directly, which is why it is tested that way.
    """
    spec, level, c_target, c_drop, c_out = PROTO, 1, 3, 1, 2
    compress = CompressionStage(spec, level, c_target, c_drop, c_out)
    decompress = DecompressionStage(spec, level, c_target, c_drop, c_out)
    with pytest.raises(ValueError, match="cannot pass through"):
        ancestor_broadcast_init_(compress, decompress)


# --------------------------------------------------------------------------
# The c_latent = 4 control arm: 4 free DOF per patch, so exactly invertible
# --------------------------------------------------------------------------


def test_full_width_latent_is_exactly_invertible_by_construction():
    """A single-stage patch over g=4 carries 4 free DOF (1 coarse mean + 3
    independent zero-mean residuals), so a 5->4->5 linear pair can be *exact*.
    Constructed analytically rather than fitted: latent = [m, r1, r2, r3], and
    the decoder recovers r4 = -(r1+r2+r3) from the conservation constraint.

    The sanity control -- if this arm cannot reconstruct, the bug is in the
    layer, not in the compression.
    """
    x = _x(PROTO, n_level0=8)
    ae = FieldSpaceAutoencoder(PROTO, n_stages=1, c_latent=4, bias=False).double()

    with torch.no_grad():
        ae.compress[0].map.weight.zero_()
        for i in range(4):
            ae.compress[0].map.weight[i, i] = 1.0  # keep [m, r1, r2, r3]

        ae.decompress[0].map.weight.zero_()
        ae.decompress[0].map.weight[0, 0] = 1.0  # m passes through
        for i in range(1, 4):
            ae.decompress[0].map.weight[i, i] = 1.0  # r1..r3 pass through
            ae.decompress[0].map.weight[i, 4] = -1.0  # r4 = -(r1 + r2 + r3)

    torch.testing.assert_close(ae(x), x)


# --------------------------------------------------------------------------
# Trainability
# --------------------------------------------------------------------------


def test_gradients_reach_every_stage():
    x = _x(TARGET, dtype=torch.float32)
    ae = FieldSpaceAutoencoder(TARGET, n_stages=2, c_latent=4)
    ((ae(x) - x) ** 2).mean().backward()
    for name, p in ae.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_training_improves_on_zero_fill():
    """The substantive claim behind the design: a learned mix at the same latent
    size beats truncation."""
    torch.manual_seed(2)
    n0, nlev = 64, 1
    base = torch.randn(n0, 1, nlev)
    x = (base + 0.3 * torch.randn(n0, 4, nlev)).reshape(-1, nlev)

    ae = FieldSpaceAutoencoder(PROTO, n_stages=1, c_latent=1)
    ae.init_as_zero_fill()
    start = ((ae(x) - x) ** 2).mean().item()
    torch.testing.assert_close(
        torch.tensor(start), ((zero_fill_baseline(x, PROTO, 1) - x) ** 2).mean()
    )

    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        loss = ((ae(x) - x) ** 2).mean()
        loss.backward()
        opt.step()

    assert loss.item() <= start, "learned bottleneck did worse than zero-fill"


# --------------------------------------------------------------------------
# Reservoir
# --------------------------------------------------------------------------


def _levels(n=4, c=2, nlev=NLEV, seed=0):
    torch.manual_seed(seed)
    return [torch.randn(n, 1, nlev), torch.randn(n * 4, c, nlev)]


def test_reservoir_is_a_ring_buffer():
    res = LatentReservoir(capacity=3)
    for i in range(5):
        res.push(_levels(seed=i), unix_seconds=float(i))
    assert len(res) == 3


def test_reservoir_rejects_bad_capacity():
    with pytest.raises(ValueError):
        LatentReservoir(capacity=0)


def test_history_returns_none_until_enough_states():
    res = LatentReservoir(capacity=8)
    res.push(_levels(), 0.0)
    assert res.history(3) is None
    res.push(_levels(seed=1), 1.0)
    res.push(_levels(seed=2), 2.0)
    assert res.history(3) is not None


def test_history_stacks_on_a_leading_time_axis_oldest_first():
    res = LatentReservoir(capacity=8)
    pushed = [_levels(seed=i) for i in range(3)]
    for i, lv in enumerate(pushed):
        res.push(lv, float(i))

    hist = res.history(3)
    assert len(hist) == 2  # one tensor per pyramid level
    assert hist[0].shape == (3, 4, 1, NLEV)
    assert hist[1].shape == (3, 16, 2, NLEV)
    for t in range(3):
        torch.testing.assert_close(hist[0][t], pushed[t][0])


def test_history_window_follows_the_most_recent_states():
    res = LatentReservoir(capacity=8)
    for i in range(5):
        res.push(_levels(seed=i), float(i))
    torch.testing.assert_close(res.history(2)[0][-1], _levels(seed=4)[0])


def test_reservoir_detaches_so_no_autograd_graph_survives():
    res = LatentReservoir(capacity=4)
    live = [lv.requires_grad_(True) for lv in _levels()]
    res.push(live, 0.0)
    for lv in res.history(1):
        assert not lv.requires_grad


def test_reservoir_stores_a_copy_not_a_view():
    res = LatentReservoir(capacity=4)
    levels = _levels()
    res.push(levels, 0.0)
    levels[0].add_(1.0)
    assert not torch.allclose(res.history(1)[0][0], levels[0])


def test_replay_is_refused_while_the_encoder_is_still_training():
    res = LatentReservoir(capacity=4, allow_replay=False)
    res.push(_levels(), 0.0)
    with pytest.raises(RuntimeError, match="staleness"):
        res.sample(1)


def test_replay_samples_once_the_encoder_is_frozen():
    res = LatentReservoir(capacity=8, allow_replay=True)
    for i in range(5):
        res.push(_levels(seed=i), float(i))
    got = res.sample(3, generator=torch.Generator().manual_seed(0))
    assert len(got) == 3
    assert len({s.unix_seconds for s in got}) == 3  # without replacement


def test_replay_refuses_to_oversample():
    res = LatentReservoir(capacity=8, allow_replay=True)
    res.push(_levels(), 0.0)
    with pytest.raises(ValueError):
        res.sample(2)


def test_reservoir_reports_what_it_costs():
    res = LatentReservoir(capacity=4)
    levels = _levels()
    res.push(levels, 0.0)
    res.push(levels, 1.0)
    assert res.numel() == 2 * sum(lv.numel() for lv in levels)


def test_clear_empties_the_reservoir():
    res = LatentReservoir(capacity=4)
    res.push(_levels(), 0.0)
    res.clear()
    assert len(res) == 0
