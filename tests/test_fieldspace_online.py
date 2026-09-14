"""Unit tests for fieldspace_online.py: OnlineFieldSpaceTrainer.

All of this runs on CPU with a small, hand-built LocalMGrid and synthetic
tensors -- no comin, no MPI, no GPU, no live ICON simulation, and no
dependency on icon_mgrid_utils.build_local_mgrid's COMIN-facing plumbing
(already covered by test_icon_mgrid_utils.py). Mirrors test_gnn_online.py's
pattern for the GNN trainer.
"""

import math

import pytest
import torch

from fieldspace_online import FieldSpaceSnapshot, OnlineFieldSpaceTrainer
from icon_mgrid_utils import LocalMGrid


def _tiled_adjc(n: int):
    """``n`` cells, each with 3 real, well-connected cyclic neighbors tiled
    into GridLayer's standard 8-column pattern (matching real ICON
    adjacency -- see icon_mgrid_utils.build_fine_adjacency's `_TILE_PATTERN`
    -- fully valid, no masking).

    An earlier version of this helper used a sparse single-link ring
    (each cell's only real neighbor at column 1, everything else masked).
    That crashed GridLayer at zoom=1: its HEALPix "polar discontinuity"
    fixup (`propagate_assignments`, columns 2 and 6) relies on each real
    neighbor slot being redundantly tiled across multiple columns (the
    real ICON convention) to find a substitute when one copy is masked; a
    single-link ring has none of that redundancy and diverges. Verified by
    direct probing (see the plan document) that real tiled adjacency, even
    fully masked at every cell, does not hit this -- so this tiled
    construction is both crash-safe and far more representative of actual
    per-rank local-patch adjacency than a toy ring would be.
    """
    tile = [0, 1, 2, 0, 1, 2, 0, 1]
    idx = torch.arange(n)
    nbr = torch.stack(
        [(idx + 1) % n, (idx + max(1, n // 5)) % n, (idx + max(2, 2 * n // 5)) % n], dim=-1
    )
    adjc = torch.cat([idx.unsqueeze(-1), nbr[:, tile]], dim=-1)
    adjc_mask = torch.zeros(n, 9, dtype=torch.bool)
    return adjc, adjc_mask


def _tiny_mgrid(n_coarse: int = 10, owned=None) -> LocalMGrid:
    """``n_coarse`` coarse tokens, 4 fine children each, both zooms built
    via `_tiled_adjc` -- enough for GridLayer/MG_Transformer to build and
    run. Grid-building correctness itself (masking, re-indexing, orphan
    handling) is exercised by test_icon_mgrid_utils.py against the real
    build_local_mgrid logic; this file only needs a valid grid to drive
    OnlineFieldSpaceTrainer's own orchestration logic.

    Default ``n_coarse=10`` is deliberately >= fieldspace_online's
    ``_MIN_ZOOM_CELLS`` (9): FieldSpaceNN's GridLayer fills masked
    adjacency slots with raw column-index values up to 8, which is out of
    bounds (a real IndexError, discovered while writing this test) for any
    zoom with fewer than 9 cells -- not just a HEALPix-only non-issue.
    """
    n_fine = 4 * n_coarse
    if owned is None:
        # First 3/4 of the fine cells owned, last 1/4 halo -- enough of
        # each to exercise the masked-loss test below.
        n_owned = (3 * n_fine) // 4
        owned = tuple(i < n_owned for i in range(n_fine))

    coarse_adjc, coarse_adjc_mask = _tiled_adjc(n_coarse)
    coarse_coords = torch.stack(
        [torch.arange(n_coarse, dtype=torch.float32) * 0.1, torch.zeros(n_coarse)], dim=-1
    )

    fine_adjc, fine_adjc_mask = _tiled_adjc(n_fine)
    fine_coords = torch.stack(
        [torch.arange(n_fine, dtype=torch.float32) * 0.05, torch.zeros(n_fine)], dim=-1
    )

    return LocalMGrid(
        n_nodes_fine=n_fine,
        n_valid_fine=n_fine,
        fine_owned_mask=torch.ones(n_fine, dtype=torch.bool),
        fine_local_ids_kept=torch.arange(n_fine, dtype=torch.int64),
        fine_adjc=fine_adjc,
        fine_adjc_mask=fine_adjc_mask,
        fine_coords=fine_coords,
        fine_compact_owned_mask=torch.tensor(owned, dtype=torch.bool),
        n_coarse=n_coarse,
        coarse_adjc=coarse_adjc,
        coarse_adjc_mask=coarse_adjc_mask,
        coarse_coords=coarse_coords,
        n_fine_kept=n_fine,
        parent_group_size_histogram={4: n_coarse},
    )


def _tiny_trainer(nlev: int = 2, n_coarse: int = 10) -> OnlineFieldSpaceTrainer:
    mgrid = _tiny_mgrid(n_coarse=n_coarse)
    return OnlineFieldSpaceTrainer(
        nlev=nlev,
        mgrid=mgrid,
        att_dim=8,
        n_head_channels=4,
        grad_clip=1.0,
        use_ddp=False,
        device=torch.device("cpu"),
        log_fn=lambda msg: None,
        rank=0,
    )


def test_forward_preserves_shape():
    torch.manual_seed(0)
    trainer = _tiny_trainer()
    n_fine, nlev = trainer.mgrid.n_fine_kept, trainer.nlev

    x = torch.randn(n_fine, nlev) * 0.1
    out = trainer._forward(x)
    assert out.shape == (n_fine, nlev)


def test_masked_mse_ignores_halo_cells():
    trainer = _tiny_trainer()
    n_fine, nlev = trainer.mgrid.n_fine_kept, trainer.nlev
    owned = trainer.mgrid.fine_compact_owned_mask
    assert owned.any() and not owned.all()  # sanity: _tiny_mgrid has both

    pred = torch.zeros(n_fine, nlev)
    target = torch.zeros(n_fine, nlev)
    target[~owned] = 100.0  # error only on halo cells -> must not move the loss
    loss = trainer._masked_mse(pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

    target2 = target.clone()
    owned_idx = torch.nonzero(owned)[0].item()
    target2[owned_idx] = 1.0  # error on an owned cell -> loss must move
    loss2 = trainer._masked_mse(pred, target2)
    assert loss2.item() > 0.0


def test_train_rollout_step_skips_on_nan_source():
    trainer = _tiny_trainer()
    n_fine, nlev = trainer.mgrid.n_fine_kept, trainer.nlev

    source = FieldSpaceSnapshot(x_fine=torch.full((n_fine, nlev), float("nan")), unix_seconds=0.0)
    target = FieldSpaceSnapshot(x_fine=torch.zeros(n_fine, nlev), unix_seconds=1.0)

    result = trainer.train_rollout_step(source, target, n_steps=2)
    assert result["skipped"] is True
    assert math.isnan(result["loss"])


def test_train_rollout_step_skips_on_nan_target():
    trainer = _tiny_trainer()
    n_fine, nlev = trainer.mgrid.n_fine_kept, trainer.nlev

    source = FieldSpaceSnapshot(x_fine=torch.zeros(n_fine, nlev), unix_seconds=0.0)
    target = FieldSpaceSnapshot(x_fine=torch.full((n_fine, nlev), float("nan")), unix_seconds=1.0)

    result = trainer.train_rollout_step(source, target, n_steps=2)
    assert result["skipped"] is True
    assert math.isnan(result["loss"])


def test_train_rollout_step_runs_and_reports_finite_metrics():
    torch.manual_seed(0)
    trainer = _tiny_trainer()
    n_fine, nlev = trainer.mgrid.n_fine_kept, trainer.nlev

    source = trainer.prepare_snapshot(torch.randn(n_fine, nlev) * 0.1, unix_seconds=0.0)
    target = trainer.prepare_snapshot(torch.randn(n_fine, nlev) * 0.1, unix_seconds=2.0)

    result = trainer.train_rollout_step(source, target, n_steps=2)
    assert result["skipped"] is False
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["grad_norm"])
    assert set(result["loss_dict"].keys()) == {"train/MSE"}
    assert result["loss_dict"]["train/MSE"] == pytest.approx(result["loss"])


def test_train_rollout_step_takes_exactly_n_forward_passes():
    trainer = _tiny_trainer()
    n_fine, nlev = trainer.mgrid.n_fine_kept, trainer.nlev

    n_forward_calls = 0
    real_forward = trainer._forward

    def counting_forward(x):
        nonlocal n_forward_calls
        n_forward_calls += 1
        return real_forward(x)

    trainer._forward = counting_forward

    source = trainer.prepare_snapshot(torch.randn(n_fine, nlev) * 0.1, unix_seconds=0.0)
    target = trainer.prepare_snapshot(torch.randn(n_fine, nlev) * 0.1, unix_seconds=3.0)
    trainer.train_rollout_step(source, target, n_steps=3)

    assert n_forward_calls == 3


def test_predict_preserves_shape():
    trainer = _tiny_trainer()
    n_fine, nlev = trainer.mgrid.n_fine_kept, trainer.nlev
    snapshot = trainer.prepare_snapshot(torch.randn(n_fine, nlev), unix_seconds=0.0)

    pred = trainer.predict(snapshot, n_steps=3)
    assert pred.shape == (n_fine, nlev)


def test_unusable_trainer_never_builds_model():
    """A rank whose coarse zoom is too small (n_coarse < _MIN_ZOOM_CELLS,
    the GridLayer out-of-bounds constraint discovered while writing this
    test suite -- see _tiny_mgrid's docstring) must never even attempt to
    build MG_Transformer/GridLayer, not just skip training calls."""
    mgrid = _tiny_mgrid(n_coarse=3)  # below _MIN_ZOOM_CELLS=9
    trainer = OnlineFieldSpaceTrainer(
        nlev=2, mgrid=mgrid, att_dim=8, n_head_channels=4,
        use_ddp=False, device=torch.device("cpu"), log_fn=lambda msg: None, rank=0,
    )
    assert trainer._usable is False
    assert trainer.model is None
    assert trainer.forward_model is None
    assert trainer.optimizer is None


def test_unusable_trainer_skips_training_and_prediction():
    """Exercise the skip branch in train_rollout_step/predict directly, via
    the `_usable` gate itself (not by mutating `mgrid.n_coarse` post-hoc,
    which the code no longer consults after construction -- see
    OnlineFieldSpaceTrainer.__init__)."""
    trainer = _tiny_trainer()
    n_fine, nlev = trainer.mgrid.n_fine_kept, trainer.nlev
    source = trainer.prepare_snapshot(torch.randn(n_fine, nlev) * 0.1, unix_seconds=0.0)
    target = trainer.prepare_snapshot(torch.randn(n_fine, nlev) * 0.1, unix_seconds=1.0)

    trainer._usable = False

    result = trainer.train_rollout_step(source, target, n_steps=2)
    assert result["skipped"] is True
    assert math.isnan(result["loss"])

    pred = trainer.predict(source, n_steps=1)
    assert torch.equal(pred, source.x_fine)
