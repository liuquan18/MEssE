"""Unit tests for fieldspace_AE_online.py: LatentReservoir,
FieldSpaceAEForecaster and OnlineFieldSpaceAETrainer.

CPU only, on the same small hand-built LocalMGrid as test_fieldspace_online.py
-- no comin, no MPI, no GPU, no live ICON simulation.
"""

import math

import pytest
import torch

from fieldspace_AE_online import LatentReservoir, OnlineFieldSpaceAETrainer
from fieldspacenn.src.modules.grids.grid_utils import decode_zooms
from test_fieldspace_online import _tiny_mgrid

NLEV = 2
N_HISTORY = 3
ROLLOUT_STEPS = 2
STEP_SECONDS = 600.0
T0 = 1_780_000_000.0  # unix seconds, mid-2026


def _tiny_trainer(n_coarse: int = 10, latent_channels: int = 4, **kwargs) -> OnlineFieldSpaceAETrainer:
    torch.manual_seed(0)
    config = dict(
        nlev=NLEV,
        mgrid=_tiny_mgrid(n_coarse=n_coarse),
        n_history=N_HISTORY,
        rollout_steps=ROLLOUT_STEPS,
        latent_channels=latent_channels,
        att_dim=8,
        n_head_channels=4,
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
    x = torch.randn(trainer.mgrid.n_fine_kept, NLEV) * 0.1
    return trainer.prepare_snapshot(x, T0 + step * STEP_SECONDS)


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


def test_pyramid_is_mean_plus_zero_mean_residual_and_decodes_exactly():
    model = _tiny_trainer().model
    x = torch.randn(model.mgrid.n_fine_kept, NLEV)
    pyr = model.to_pyramid(x)

    n_coarse = model.mgrid.n_coarse
    group_mean = pyr[1].reshape(n_coarse, 4, NLEV).mean(dim=1)
    torch.testing.assert_close(group_mean, torch.zeros_like(group_mean), atol=1e-6, rtol=0)

    back = decode_zooms(pyr, model._sample_configs_step, out_zoom=1)[1]
    torch.testing.assert_close(back.reshape(-1, NLEV), x, atol=1e-6, rtol=0)


def test_to_pyramid_does_not_modify_its_input():
    model = _tiny_trainer().model
    x = torch.randn(model.mgrid.n_fine_kept, NLEV)
    x_before = x.clone()
    model.to_pyramid(x)
    assert torch.equal(x, x_before)


def test_latent_lives_on_the_coarse_zoom_with_latent_channels():
    model = _tiny_trainer(latent_channels=2).model
    x = torch.randn(model.mgrid.n_fine_kept, NLEV)
    latent = model.encode(x, T0)
    assert list(latent.keys()) == [0]
    assert latent[0].shape == (1, 1, 1, model.mgrid.n_coarse, NLEV, 2)
    assert latent[0].numel() / x.numel() == pytest.approx(2 / 4)
    assert model.decode(latent, T0).shape == x.shape


def test_untrained_processor_is_persistence_of_the_latest_latent():
    model = _tiny_trainer().model
    torch.manual_seed(1)
    window = {0: torch.randn(1, 1, N_HISTORY, model.mgrid.n_coarse, NLEV, 4)}
    seconds = [T0 + i * STEP_SECONDS for i in range(N_HISTORY)]
    nxt = model.step(window, seconds, seconds[-1] + STEP_SECONDS)
    torch.testing.assert_close(nxt[0], window[0][:, :, -1:], atol=1e-6, rtol=0)


def test_model_time_changes_the_output():
    """The ICON time really reaches the network. Gates start at ~1e-12, which
    hides every update at initialization, so perturb all weights first."""
    model = _tiny_trainer().model
    torch.manual_seed(2)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    x = torch.randn(model.mgrid.n_fine_kept, NLEV)
    noon, midnight = T0 + 0.5 * 86400.0, T0
    assert not torch.allclose(model.encode(x, noon)[0], model.encode(x, midnight)[0])

    window = {0: torch.randn(1, 1, N_HISTORY, model.mgrid.n_coarse, NLEV, 4)}
    a = model.rollout(window, [noon + i for i in range(N_HISTORY)], 1, 1.0)
    b = model.rollout(window, [midnight + i for i in range(N_HISTORY)], 1, 1.0)
    assert not torch.allclose(a[0], b[0])


def test_every_parameter_gets_a_gradient_in_one_training_pass():
    """DDP is built with find_unused_parameters=False, which requires this."""
    model = _tiny_trainer().model
    x = torch.randn(model.mgrid.n_fine_kept, NLEV)
    window = {0: torch.randn(1, 1, N_HISTORY, model.mgrid.n_coarse, NLEV, 4)}
    seconds = [T0 + i * STEP_SECONDS for i in range(N_HISTORY)]
    now = seconds[-1] + ROLLOUT_STEPS * STEP_SECONDS
    recon, pred, _ = model(x, now, window, seconds, ROLLOUT_STEPS, STEP_SECONDS)
    ((recon - x) ** 2).mean().add(((pred - x) ** 2).mean()).backward()
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
        recon, pred, latent = real_forward(*args, **kwargs)
        captured["pred"] = pred.detach().clone()
        return recon, pred, latent

    trainer.model.forward = capturing_forward
    assert trainer.train_step(_snapshot(trainer, 3))["skipped"] is False
    torch.testing.assert_close(captured["pred"], written_back)


def test_nan_snapshot_is_not_cached_and_restarts_history():
    trainer = _tiny_trainer()
    for step in range(2):
        trainer.train_step(_snapshot(trainer, step))
    bad = trainer.prepare_snapshot(torch.full((trainer.mgrid.n_fine_kept, NLEV), float("nan")), T0)
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
    assert pred.shape == (trainer.mgrid.n_fine_kept, NLEV)
    assert torch.isfinite(pred).all()


def test_masked_mse_ignores_halo_cells():
    trainer = _tiny_trainer()
    owned = trainer.mgrid.fine_compact_owned_mask
    assert owned.any() and not owned.all()
    pred = torch.zeros(trainer.mgrid.n_fine_kept, NLEV)
    target = torch.zeros_like(pred)
    target[~owned] = 100.0
    assert trainer._masked_mse(pred, target).item() == pytest.approx(0.0, abs=1e-6)
    target[torch.nonzero(owned)[0].item()] = 1.0
    assert trainer._masked_mse(pred, target).item() > 0.0


def test_unusable_trainer_never_builds_model_and_skips_everything():
    trainer = _tiny_trainer(n_coarse=3)  # below _MIN_ZOOM_CELLS=9
    assert trainer._usable is False
    assert trainer.model is None and trainer.forward_model is None and trainer.optimizer is None
    snap = trainer.prepare_snapshot(torch.zeros(trainer.mgrid.n_fine_kept, NLEV), T0)
    assert trainer.train_step(snap)["skipped"] is True
    assert trainer.predict() is None
    assert len(trainer.reservoir) == 0


def test_rejects_non_positive_history_or_rollout():
    with pytest.raises(ValueError):
        OnlineFieldSpaceAETrainer(nlev=NLEV, mgrid=_tiny_mgrid(), n_history=0, device=torch.device("cpu"))
    with pytest.raises(ValueError):
        OnlineFieldSpaceAETrainer(nlev=NLEV, mgrid=_tiny_mgrid(), rollout_steps=0, device=torch.device("cpu"))
