"""Tests for fieldspace_RF_online.py on a small synthetic nested patch.

Levels 0, 2, 3, 4 (backbone 0, level 1 has a grid layer but no data), latent
level 2: two compression stages. CPU only, no comin, no MPI.
"""

import pytest
import torch

from fieldspace_RF_online import LatentReplay, OnlineReconstructorForecaster, autoencoder_configs, _Blocks
from icon_nested_mgrid import patch_mgrid
from test_icon_nested_mgrid import synthetic_global_mgrid

LEVELS, LATENT_LEVEL, N_HISTORY, NLEV = [0, 2, 3, 4], 2, 3, 1
T0, DT = 1.78e9, 600.0
N_BACKBONE = 10
N_FINE = N_BACKBONE * 4**4


def _trainer(**kwargs):
    torch.manual_seed(0)
    mgrids = patch_mgrid(synthetic_global_mgrid(n_levels=5), torch.arange(N_BACKBONE))
    config = dict(n_history=N_HISTORY, device=torch.device("cpu"), att_dim=16, n_head_channels=4,
                  hidden_dim=16, time_embed_dim=8, n_blocks=1, n_forecaster_blocks=1)
    config.update(kwargs)
    return OnlineReconstructorForecaster(mgrids, LEVELS, LATENT_LEVEL, NLEV, **config)


def _field(k):
    torch.manual_seed(100 + k)
    return torch.randn(N_FINE, NLEV)


def test_levels_are_backbone_mean_plus_zero_mean_residuals_and_invert_exactly():
    model = _trainer().model
    x = _field(0)
    x_copy = x.clone()
    levels = model.to_levels(x)
    assert torch.equal(x, x_copy)  # input untouched (encode_zooms works in place)
    assert sorted(levels) == [0, 2, 3, 4]
    assert torch.allclose(levels[0].flatten(), x.reshape(N_BACKBONE, -1).mean(dim=1), atol=1e-6)
    for parent, zoom in [(0, 2), (2, 3), (3, 4)]:
        residual = levels[zoom].reshape(-1, 4 ** (zoom - parent))
        assert torch.allclose(residual.mean(dim=1), torch.zeros(1), atol=1e-5)
    assert torch.allclose(model.to_field(levels), x, atol=1e-5)


@pytest.mark.parametrize("latent_zoom, latent", [(2, {0: 1, 2: 4}), (3, {0: 1, 2: 1, 3: 4})])
def test_latent_levels_and_channels(latent_zoom, latent):
    *_, latent_features = autoencoder_configs([0, 2, 3, 4], latent_zoom, 4, 1, _Blocks(16, 4, 16, 8))
    assert latent_features == latent


def _train(trainer, n_steps):
    """``n_steps`` sample steps; returns the stats of the last one."""
    for k in range(n_steps):
        stats = trainer.train_step(_field(k), T0 + k * DT)
    return stats


def test_history_fills_before_training_and_forecasting():
    trainer = _trainer()
    for k in range(N_HISTORY):
        assert trainer.train_step(_field(k), T0 + k * DT) == {}
    assert trainer.predict_increment(T0 + N_HISTORY * DT) is not None
    stats = trainer.train_step(_field(N_HISTORY), T0 + N_HISTORY * DT)
    assert set(stats) == {"recon", "state_ms", "incre", "persist_mse"}
    assert all(sorted(parts) == [0, 2, 3, 4] for parts in stats.values())


def test_untrained_forecast_is_persistence():
    trainer = _trainer()
    for k in range(N_HISTORY):
        trainer.train_step(_field(k), T0 + k * DT)
    increment = trainer.predict_increment(T0 + N_HISTORY * DT)
    assert increment.shape == (N_FINE, NLEV)
    assert increment.abs().max() < 1e-4


@pytest.mark.parametrize("replay", [{}, {"replay_size": 4, "replay_batch": 2}])
def test_untrained_increment_loss_is_the_persistence_mse(replay):
    """An untrained forecaster forecasts zero change, so its increment loss
    per level is the logged persistence reference, also over a replay batch."""
    stats = _train(_trainer(lr_forecast=0.0, **replay), N_HISTORY + 3)
    for zoom, value in stats["incre"].items():
        assert value == pytest.approx(stats["persist_mse"][zoom], rel=1e-3), zoom


def test_models_are_trained_independently():
    """The forecast loss never reaches the reconstructor, the reconstruction
    loss never reaches the forecaster, and the frozen copy gets no gradient."""
    trainer = _trainer()
    for k in range(N_HISTORY):
        trainer.train_step(_field(k), T0 + k * DT)
    model = trainer.model
    history, seconds = trainer._history()
    levels = model.to_levels(_field(9))
    recon, forecast = model(levels, T0 + N_HISTORY * DT, history, [seconds + [T0 + N_HISTORY * DT]])

    sum(f.square().sum() for f in forecast.values()).backward(retain_graph=True)
    assert all(p.grad is None for p in model.reconstructor.parameters())
    assert all(p.grad is None for p in model.frozen.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.forecaster.parameters())

    model.zero_grad(set_to_none=True)
    sum(r.square().sum() for r in recon.values()).backward()
    assert all(p.grad is None for p in model.forecaster.parameters())


@pytest.mark.parametrize("replay", [{}, {"replay_size": 4}])
def test_every_trainable_parameter_gets_a_gradient_in_one_pass(replay):
    """DDP (find_unused_parameters=False) needs this, with and without replay."""
    trainer = _trainer(**replay)
    _train(trainer, N_HISTORY + 2)
    seen = {}
    for name, p in trainer.model.named_parameters():
        if p.requires_grad:
            p.register_hook(lambda g, name=name: seen.__setitem__(name, True))
    trainer.train_step(_field(N_HISTORY + 2), T0 + (N_HISTORY + 2) * DT)
    trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    assert trainable and sorted(seen) == sorted(trainable)


def test_frozen_copy_follows_the_reconstructor_slowly():
    trainer = _trainer(ema_decay=0.9)
    for k in range(N_HISTORY + 3):
        trainer.train_step(_field(k), T0 + k * DT)
    live = torch.cat([p.flatten() for p in trainer.model.reconstructor.parameters()])
    frozen = torch.cat([p.flatten() for p in trainer.model.frozen.module.parameters()])
    assert int(trainer.model.frozen.n_averaged) == 3
    assert not torch.equal(live, frozen)
    assert (live - frozen).abs().max() < 0.1


def test_latent_replay_keeps_the_last_windows_and_draws_without_replacement():
    replay = LatentReplay(size=5, batch=3, generator=torch.Generator().manual_seed(0))
    assert replay.sample() == []
    for k in range(8):
        replay.add(k)  # the buffer does not look inside a window
    assert list(replay.windows) == [3, 4, 5, 6, 7]
    for _ in range(20):
        drawn = replay.sample()
        assert len(drawn) == len(set(drawn)) == 3 and set(drawn) <= {3, 4, 5, 6, 7}
    small = LatentReplay(size=5, batch=3, generator=torch.Generator().manual_seed(0))
    small.add(0)
    assert small.sample() == [0]


def test_replay_off_draws_no_random_numbers():
    rng = torch.Generator().manual_seed(0)
    state = rng.get_state()
    replay = LatentReplay(size=0, batch=10, generator=rng)
    replay.add(0)
    assert replay.sample() == [] and torch.equal(rng.get_state(), state)


def test_batched_forecast_matches_the_single_window_forecast():
    trainer = _trainer()
    _train(trainer, N_HISTORY)
    model = trainer.model
    history, seconds = trainer._history()
    times = seconds + [T0 + N_HISTORY * DT]
    single = model.forecast(history, [times])
    other = {z: torch.randn_like(h) for z, h in history.items()}
    batch = model.forecast({z: torch.cat([history[z], other[z], other[z]]) for z in history},
                           [times, [s + 3600 for s in times], [s + 7200 for s in times]])
    for z in single:
        assert batch[z].shape[0] == 3
        assert torch.allclose(batch[z][:1], single[z], atol=1e-5), z


def test_replay_starts_with_the_second_update():
    """The first update has an empty buffer, so it equals training without
    replay; every update stores its window."""
    off, on = _trainer(), _trainer(replay_size=4, replay_batch=2)
    for k in range(N_HISTORY + 2):
        s_off, s_on = off.train_step(_field(k), T0 + k * DT), on.train_step(_field(k), T0 + k * DT)
        if k == N_HISTORY:
            assert s_on == s_off
    assert s_on != s_off
    assert len(on.replay.windows) == 2 and len(off.replay.windows) == 0


def test_forecaster_batch_is_the_new_window_plus_the_drawn_ones():
    """With a draw as large as the buffer, the batch of the 3rd update is
    exactly the 3 windows stored afterwards, so its persistence reference is
    their mean."""
    trainer = _trainer(replay_size=10, replay_batch=10)
    stats = _train(trainer, N_HISTORY + 3)
    windows = trainer.replay.windows
    assert len(windows) == 3
    for z, value in stats["persist_mse"].items():
        expected = sum(float(w.increment[z].square().mean()) for w in windows) / 3
        assert value == pytest.approx(expected, rel=1e-5), z


def test_stored_window_holds_the_history_latents_increment_and_times():
    trainer = _trainer(replay_size=4)
    _train(trainer, N_HISTORY)
    history, _ = trainer._history()
    trainer.train_step(_field(N_HISTORY), T0 + N_HISTORY * DT)
    (window,) = trainer.replay.windows
    assert window.seconds == [T0 + k * DT for k in range(N_HISTORY + 1)]
    assert all(torch.equal(window.history[z], history[z]) for z in history)
    x_t, x_prev = trainer.model.to_levels(_field(N_HISTORY)), trainer.model.to_levels(_field(N_HISTORY - 1))
    assert all(torch.allclose(window.increment[z], x_t[z] - x_prev[z]) for z in x_t)
