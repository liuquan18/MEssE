"""Unit tests for fieldspace_AE_online.py: LatentReservoir,
FieldSpaceAEForecaster and OnlineFieldSpaceAETrainer.

CPU only, on a small synthetic nested patch (test_icon_nested_mgrid's grids)
with levels 0..3: backbone 0, residual levels 2 and 3 (level 1 has a grid
layer but no data), latent level 2 -- one compression stage and a skipped
level. No comin, no MPI, no GPU.
"""

import math

import numpy as np
import pytest
import torch

from fieldspace_AE_online import LatentReservoir, OnlineFieldSpaceAETrainer, compression_stages
from fieldspacenn.src.modules.grids.grid_utils import decode_zooms
from icon_nested_mgrid import build_nested_mgrid
from test_icon_nested_mgrid import _synthetic_grids

NLEV = 2
N_HISTORY = 3
ROLLOUT_STEPS = 2
STEP_SECONDS = 600.0
T0 = 1_780_000_000.0  # unix seconds, mid-2026
LEVELS = [0, 2, 3]
LATENT_LEVEL = 2


def _tiny_mgrid(n_backbone: int = 10, n_levels: int = 4):
    return build_nested_mgrid(_synthetic_grids(n_levels=n_levels), np.arange(n_backbone))


def _tiny_trainer(levels=LEVELS, latent_level=LATENT_LEVEL, latent_channels: int = 4, n_backbone: int = 10, **kwargs):
    torch.manual_seed(0)
    config = dict(
        nlev=NLEV,
        mgrid=_tiny_mgrid(n_backbone=n_backbone, n_levels=max(levels) + 1),
        levels=levels,
        latent_level=latent_level,
        n_history=N_HISTORY,
        rollout_steps=ROLLOUT_STEPS,
        latent_channels=latent_channels,
        att_dim=8,
        n_head_channels=4,
        hidden_dim=16,
        time_embed_dim=8,
        use_ddp=False,
        device=torch.device("cpu"),
        log_fn=lambda msg: None,
        rank=0,
    )
    config.update(kwargs)
    return OnlineFieldSpaceAETrainer(**config)


def _snapshot(trainer, step: int):
    torch.manual_seed(step)
    x = torch.randn(trainer.mgrid.n_fine, NLEV) * 0.1
    return trainer.prepare_snapshot(x, T0 + step * STEP_SECONDS)


def _random_window(model):
    return {
        zoom: torch.randn(1, 1, N_HISTORY, model.mgrid.n_cells(zoom), NLEV, model.latent_features[zoom])
        for zoom in model.latent_zooms
    }


def _latent(value: float, n: int = 5):
    return {0: torch.full((1, 1, 1, n, NLEV, 4), float(value))}


# --------------------------------------------------------------------------
# Reservoir
# --------------------------------------------------------------------------


def test_reservoir_is_a_ring_buffer():
    res = LatentReservoir(capacity=3)
    for i in range(5):
        res.push(_latent(i), float(i))
    assert len(res) == 3
    w = res.window(3)
    assert w.latent[0][0, 0, :, 0, 0, 0].tolist() == [2.0, 3.0, 4.0]
    assert w.unix_seconds == [2.0, 3.0, 4.0]
    assert res.latest_seconds == 4.0


def test_reservoir_rejects_bad_capacity():
    with pytest.raises(ValueError):
        LatentReservoir(capacity=0)


def test_window_is_none_until_enough_states_including_lag():
    res = LatentReservoir(capacity=8)
    assert res.latest_seconds is None
    for i in range(3):
        res.push(_latent(i), float(i))
    assert res.window(3) is not None
    assert res.window(4) is None
    assert res.window(2, lag=1) is not None
    assert res.window(3, lag=1) is None


def test_window_stacks_on_time_axis_oldest_first_and_honours_lag():
    res = LatentReservoir(capacity=8)
    for i in range(5):
        res.push(_latent(i), 100.0 + i)
    w = res.window(3, lag=1)
    assert w.latent[0].shape == (1, 1, 3, 5, NLEV, 4)
    assert w.latent[0][0, 0, :, 0, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert w.unix_seconds == [101.0, 102.0, 103.0]


def test_reservoir_stores_a_detached_copy():
    res = LatentReservoir(capacity=4)
    live = {0: torch.zeros(1, 1, 1, 5, NLEV, 4, requires_grad=True)}
    res.push(live, 0.0)
    with torch.no_grad():
        live[0].add_(1.0)
    stored = res.window(1).latent[0]
    assert not stored.requires_grad
    assert torch.equal(stored, torch.zeros_like(stored))


def test_reservoir_numel_and_clear():
    res = LatentReservoir(capacity=4)
    res.push(_latent(0), 0.0)
    res.push(_latent(1), 1.0)
    assert res.numel() == 2 * _latent(0)[0].numel()
    res.clear()
    assert len(res) == 0 and res.window(1) is None and res.latest_seconds is None


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def test_compression_stages_fold_finest_first_down_to_latent_level():
    assert compression_stages([4, 6, 7, 8], 6) == [(8, 7), (7, 6)]
    assert compression_stages([4, 6, 7, 8], 7) == [(8, 7)]
    assert compression_stages([4, 6, 7, 8], 8) == []
    assert compression_stages([4, 6, 8], 6) == [(8, 6)]


def test_model_rejects_inconsistent_levels():
    with pytest.raises(ValueError, match="latent_level"):
        _tiny_trainer(latent_level=0)
    with pytest.raises(ValueError, match="must start at"):
        _tiny_trainer(levels=[1, 2, 3], latent_level=2)


def test_pyramid_is_backbone_mean_plus_zero_mean_residuals_and_decodes_exactly():
    model = _tiny_trainer().model
    x = torch.randn(model.mgrid.n_fine, NLEV)
    pyr = model.to_pyramid(x)
    assert sorted(pyr) == LEVELS

    n_backbone = model.mgrid.n_backbone
    torch.testing.assert_close(
        pyr[0].reshape(n_backbone, NLEV), x.reshape(n_backbone, 64, NLEV).mean(dim=1), atol=1e-6, rtol=0
    )
    # Each residual level averages to zero over the children of its model parent.
    for zoom, parent in ((2, 0), (3, 2)):
        group_mean = pyr[zoom].reshape(n_backbone * 4**parent, 4 ** (zoom - parent), NLEV).mean(dim=1)
        torch.testing.assert_close(group_mean, torch.zeros_like(group_mean), atol=1e-6, rtol=0)

    back = decode_zooms(pyr, model._sample_configs_step, out_zoom=3)[3]
    torch.testing.assert_close(back.reshape(-1, NLEV), x, atol=1e-5, rtol=0)


def test_to_pyramid_does_not_modify_its_input():
    model = _tiny_trainer().model
    x = torch.randn(model.mgrid.n_fine, NLEV)
    x_before = x.clone()
    model.to_pyramid(x)
    assert torch.equal(x, x_before)


@pytest.mark.parametrize(
    "levels, latent_level, latent_features",
    [([0, 2, 3], 2, {0: 1, 2: 2}), ([0, 1, 2, 3], 1, {0: 1, 1: 2}), ([0, 2, 3], 3, {0: 1, 2: 1, 3: 1})],
)
def test_latent_zooms_channels_and_size(levels, latent_level, latent_features):
    model = _tiny_trainer(levels=levels, latent_level=latent_level, latent_channels=2).model
    x = torch.randn(model.mgrid.n_fine, NLEV)
    latent = model.encode(x, T0)
    assert model.latent_features == latent_features
    assert sorted(latent) == model.latent_zooms == sorted(latent_features)
    for zoom, features in latent_features.items():
        assert latent[zoom].shape == (1, 1, 1, model.mgrid.n_cells(zoom), NLEV, features)
    assert sum(t.numel() for t in latent.values()) == model.latent_numel()
    assert model.decode(latent, T0).shape == x.shape


def test_untrained_processor_is_persistence_of_the_latest_latent():
    model = _tiny_trainer().model
    torch.manual_seed(1)
    window = _random_window(model)
    seconds = [T0 + i * STEP_SECONDS for i in range(N_HISTORY)]
    nxt = model.step(window, seconds, seconds[-1] + STEP_SECONDS)
    for zoom in model.latent_zooms:
        torch.testing.assert_close(nxt[zoom], window[zoom][:, :, -1:], atol=1e-6, rtol=0)


def test_model_time_changes_the_output():
    """The ICON time really reaches the network. Gates start at ~1e-12, which
    hides every update at initialization, so perturb all weights first."""
    model = _tiny_trainer().model
    torch.manual_seed(2)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    x = torch.randn(model.mgrid.n_fine, NLEV)
    noon, midnight = T0 + 0.5 * 86400.0, T0
    assert not torch.allclose(model.encode(x, noon)[2], model.encode(x, midnight)[2])

    window = _random_window(model)
    a = model.rollout(window, [noon + i for i in range(N_HISTORY)], 1, 1.0)
    b = model.rollout(window, [midnight + i for i in range(N_HISTORY)], 1, 1.0)
    assert not torch.allclose(a[2], b[2])


@pytest.mark.parametrize("levels, latent_level", [([0, 2, 3], 2), ([0, 1, 2, 3], 1)])
def test_every_parameter_gets_a_gradient_in_one_training_pass(levels, latent_level):
    """DDP is built with find_unused_parameters=False, which requires this."""
    model = _tiny_trainer(levels=levels, latent_level=latent_level).model
    x = torch.randn(model.mgrid.n_fine, NLEV)
    window = _random_window(model)
    seconds = [T0 + i * STEP_SECONDS for i in range(N_HISTORY)]
    now = seconds[-1] + ROLLOUT_STEPS * STEP_SECONDS
    recon, pred, _, prev_recon = model(x, now, window, seconds, ROLLOUT_STEPS, STEP_SECONDS)
    # The trainer's loss: reconstruction plus the predicted-change term.
    x_prev = torch.randn_like(x)
    ((recon - x) ** 2).mean().add((((pred - prev_recon) - (x - x_prev)) ** 2).mean()).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


def test_warm_up_caches_without_training_then_trains():
    trainer = _tiny_trainer()
    n_warm = N_HISTORY + ROLLOUT_STEPS - 1
    for step in range(n_warm):
        result = trainer.train_step(_snapshot(trainer, step))
        assert result["skipped"] is True
        assert result["needs_rollback"] is False
        assert len(trainer.reservoir) == step + 1

    result = trainer.train_step(_snapshot(trainer, n_warm))
    assert result["skipped"] is False
    assert math.isfinite(result["loss"]) and math.isfinite(result["grad_norm"])
    losses = result["loss_dict"]
    assert set(losses) == {"train/MSE_pred", "train/MSE_recon"}
    assert result["loss"] == pytest.approx(losses["train/MSE_pred"] + losses["train/MSE_recon"], rel=1e-5)
    assert len(trainer.reservoir) == trainer.reservoir.capacity
    assert trainer.reservoir.latest_seconds == T0 + n_warm * STEP_SECONDS


def test_step_length_is_measured_from_snapshot_times():
    trainer = _tiny_trainer()
    trainer.train_step(_snapshot(trainer, 0))
    assert trainer.step_seconds is None
    trainer.train_step(_snapshot(trainer, 1))
    assert trainer.step_seconds == STEP_SECONDS


def test_training_rolls_the_processor_exactly_rollout_steps_times():
    trainer = _tiny_trainer()
    for step in range(N_HISTORY + ROLLOUT_STEPS - 1):
        trainer.train_step(_snapshot(trainer, step))

    calls = []
    real_step = trainer.model.step

    def recording_step(window, window_seconds, next_seconds):
        calls.append(next_seconds)
        return real_step(window, window_seconds, next_seconds)

    trainer.model.step = recording_step
    target_step = N_HISTORY + ROLLOUT_STEPS - 1
    assert trainer.train_step(_snapshot(trainer, target_step))["skipped"] is False
    # The rollout lands exactly on the target's time.
    assert calls == [T0 + (target_step - ROLLOUT_STEPS + k) * STEP_SECONDS for k in (1, 2)]


def test_design_prediction_scored_at_t_is_the_one_written_back_at_t_minus_1():
    """n_history=2, rollout=1: after step t-1 the plugin writes predict() to
    ICON; at step t the loss must score exactly that prediction against x_t."""
    trainer = _tiny_trainer(n_history=2, rollout_steps=1)
    for step in range(3):  # steps 0-1 only fill the reservoir, step 2 trains
        trainer.train_step(_snapshot(trainer, step))
    written_back = trainer.predict(n_steps=1)  # forecast for step 3

    captured = {}
    real_forward = trainer.model.forward

    def capturing_forward(*args, **kwargs):
        recon, pred, latent, prev_recon = real_forward(*args, **kwargs)
        captured["pred"] = pred.detach().clone()
        return recon, pred, latent, prev_recon

    trainer.model.forward = capturing_forward
    assert trainer.train_step(_snapshot(trainer, 3))["skipped"] is False
    torch.testing.assert_close(captured["pred"], written_back)


def test_nan_snapshot_is_not_cached_and_restarts_history():
    trainer = _tiny_trainer()
    for step in range(2):
        trainer.train_step(_snapshot(trainer, step))
    bad = trainer.prepare_snapshot(torch.full((trainer.mgrid.n_fine, NLEV), float("nan")), T0)
    result = trainer.train_step(bad)
    assert result["skipped"] is True
    assert math.isnan(result["loss"])
    assert len(trainer.reservoir) == 0


def test_predict_needs_n_history_latents():
    trainer = _tiny_trainer()
    for step in range(N_HISTORY - 1):
        trainer.train_step(_snapshot(trainer, step))
        assert trainer.predict() is None
    trainer.train_step(_snapshot(trainer, N_HISTORY - 1))
    pred = trainer.predict(n_steps=2)
    assert pred.shape == (trainer.mgrid.n_fine, NLEV)
    assert torch.isfinite(pred).all()


def test_rejects_patch_below_gridlayer_minimum():
    with pytest.raises(ValueError, match="at least 9"):
        _tiny_trainer(n_backbone=8)


def test_rejects_non_positive_history_or_rollout():
    with pytest.raises(ValueError):
        _tiny_trainer(n_history=0)
    with pytest.raises(ValueError):
        _tiny_trainer(rollout_steps=0)


def test_predict_increment_is_the_forecast_minus_the_latest_reconstruction():
    """What the plugin writes to var_predict: the predicted change, with both
    decodes taken at the current weights so the autoencoder error cancels."""
    trainer = _tiny_trainer(n_history=2, rollout_steps=1)
    assert trainer.predict_increment() is None  # nothing cached yet
    for step in range(3):
        trainer.train_step(_snapshot(trainer, step))

    increment = trainer.predict_increment(n_steps=1)
    forecast = trainer.predict(n_steps=1)
    window = trainer.reservoir.window(trainer.n_history)
    latest = {zoom: w[:, :, -1:] for zoom, w in window.latent.items()}
    with torch.no_grad():
        reference = trainer.model.decode(latest, window.unix_seconds[-1])
    torch.testing.assert_close(increment, forecast - reference, atol=1e-6, rtol=0)


def test_prediction_loss_scores_the_change_not_the_absolute_field():
    """The reported prediction loss must be the increment MSE the plugin's
    fc_rmse measures, not the MSE against the absolute field."""
    trainer = _tiny_trainer(n_history=2, rollout_steps=1)
    snapshots = [_snapshot(trainer, step) for step in range(4)]
    for snapshot in snapshots[:3]:
        trainer.train_step(snapshot)

    captured = {}
    real_forward = trainer.model.forward

    def capturing_forward(*args, **kwargs):
        recon, pred, latent, prev_recon = real_forward(*args, **kwargs)
        captured["pred"] = pred.detach().clone()
        captured["prev_recon"] = prev_recon.detach().clone()
        return recon, pred, latent, prev_recon

    trainer.model.forward = capturing_forward
    result = trainer.train_step(snapshots[3])
    assert result["skipped"] is False

    x, x_prev = snapshots[3].x_fine, snapshots[2].x_fine  # window ends one step back
    increment_mse = (((captured["pred"] - captured["prev_recon"]) - (x - x_prev)) ** 2).mean()
    absolute_mse = ((captured["pred"] - x) ** 2).mean()
    assert result["loss_dict"]["train/MSE_pred"] == pytest.approx(increment_mse.item(), rel=1e-5)
    assert result["loss_dict"]["train/MSE_pred"] != pytest.approx(absolute_mse.item(), rel=1e-3)


def test_history_fields_are_cleared_with_the_latents():
    trainer = _tiny_trainer()
    for step in range(2):
        trainer.train_step(_snapshot(trainer, step))
    assert len(trainer._fields) == 2
    trainer.train_step(trainer.prepare_snapshot(torch.full((trainer.mgrid.n_fine, NLEV), float("nan")), T0))
    assert len(trainer.reservoir) == 0 and len(trainer._fields) == 0
