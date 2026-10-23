"""Tests for fieldspace_RF_online.py on a small synthetic nested patch.

Levels 0, 2, 3, 4 (backbone 0, level 1 has a grid layer but no data), latent
level 2: two compression stages. Variable 0 plus two forcings unless a test
says otherwise. CPU only, no comin, no MPI.
"""

import pytest
import torch

from fieldspace_RF_online import ReplayBuffer, OnlineReconstructorForecaster, autoencoder_configs, _BlockConfigs
from icon_nested_mgrid import patch_mgrid
from test_icon_nested_mgrid import synthetic_global_mgrid

LEVELS, LATENT_LEVEL, N_CONTEXT, NLEV = [0, 2, 3, 4], 2, 3, 1
N_VARIABLES = 3
T0, DT = 1.78e9, 600.0
N_BACKBONE = 10
N_FINE = N_BACKBONE * 4**4


def _trainer(n_variables=N_VARIABLES, **kwargs):
    torch.manual_seed(0)
    mgrids = patch_mgrid(synthetic_global_mgrid(n_levels=5), torch.arange(N_BACKBONE))
    config = dict(n_context=N_CONTEXT, device=torch.device("cpu"), att_dim=16, n_head_channels=4,
                  hidden_dim=16, time_embed_dim=8, n_blocks=1, n_forecaster_blocks=1, n_variables=n_variables)
    config.update(kwargs)
    return OnlineReconstructorForecaster(mgrids, LEVELS, LATENT_LEVEL, NLEV, **config)


def _field(k, n_variables=N_VARIABLES):
    torch.manual_seed(100 + k)
    return torch.randn(N_FINE, n_variables, NLEV)


def test_levels_are_backbone_mean_plus_zero_mean_residuals_and_invert_exactly():
    model = _trainer().model
    x = _field(0)
    x_copy = x.clone()
    levels = model.to_levels(x)
    assert torch.equal(x, x_copy)  # input untouched (encode_zooms works in place)
    assert sorted(levels) == [0, 2, 3, 4]
    per_variable = x.permute(1, 0, 2).reshape(N_VARIABLES, N_BACKBONE, -1)
    assert torch.allclose(levels[0][0, :, 0, :, 0, 0], per_variable.mean(dim=2), atol=1e-6)
    for parent, zoom in [(0, 2), (2, 3), (3, 4)]:
        residual = levels[zoom].reshape(N_VARIABLES, -1, 4 ** (zoom - parent))
        assert torch.allclose(residual.mean(dim=2), torch.zeros(1), atol=1e-5)
    assert torch.allclose(model.to_field(levels), x, atol=1e-5)


@pytest.mark.parametrize("latent_zoom, latent", [(2, {0: 1, 2: 4}), (3, {0: 1, 2: 1, 3: 4})])
def test_latent_levels_and_channels(latent_zoom, latent):
    *_, latent_features = autoencoder_configs([0, 2, 3, 4], latent_zoom, 4, 1, _BlockConfigs(16, 4, 16, 8))
    assert latent_features == latent


def _train(trainer, n_steps):
    """``n_steps`` sample steps; returns the stats of the last one."""
    for k in range(n_steps):
        stats = trainer.train_step(_field(k), T0 + k * DT)
    return stats


def test_context_fills_before_training_and_forecasting():
    trainer = _trainer()
    for k in range(N_CONTEXT):
        assert trainer.train_step(_field(k), T0 + k * DT) == {}
    assert trainer.predict_increment(T0 + N_CONTEXT * DT) is not None
    stats = trainer.train_step(_field(N_CONTEXT), T0 + N_CONTEXT * DT)
    assert set(stats) == {"recon", "state_ms", "latent_mse", "latent_persist"}
    assert sorted(stats["recon"]) == sorted(stats["state_ms"]) == [0, 2, 3, 4]
    assert sorted(stats["latent_mse"]) == sorted(stats["latent_persist"]) == [0, 2]  # latent levels


@pytest.mark.parametrize("n_variables", [1, N_VARIABLES])
def test_untrained_forecast_is_persistence(n_variables):
    trainer = _trainer(n_variables)
    for k in range(N_CONTEXT):
        trainer.train_step(_field(k, n_variables), T0 + k * DT)
    increment = trainer.predict_increment(T0 + N_CONTEXT * DT)
    assert increment.shape == (N_FINE, NLEV)
    assert increment.abs().max() < 1e-4


@pytest.mark.parametrize("replay", [{}, {"replay_size": 4, "replay_batch": 2}])
def test_untrained_latent_loss_is_the_persistence_mse(replay):
    """An untrained forecaster returns the newest context latent, so its
    latent loss per level is the logged persistence reference, also over a
    replay batch."""
    stats = _train(_trainer(lr_forecast=0.0, **replay), N_CONTEXT + 3)
    for zoom, value in stats["latent_mse"].items():
        assert value == pytest.approx(stats["latent_persist"][zoom], rel=1e-3), zoom


def test_models_are_trained_independently():
    """The forecast loss never reaches the reconstructor, the reconstruction
    loss never reaches the forecaster, and the frozen copy gets no gradient."""
    trainer = _trainer()
    for k in range(N_CONTEXT):
        trainer.train_step(_field(k), T0 + k * DT)
    model = trainer.model
    context, seconds = trainer.replay_buffer.context()
    levels = model.to_levels(_field(9))
    recon, forecast = model(levels, T0 + N_CONTEXT * DT, context, [seconds + [T0 + N_CONTEXT * DT]])
    assert all(r.shape[1] == N_VARIABLES for r in recon.values())  # every variable is reconstructed
    assert all(f.shape[1] == 1 for f in forecast.values())  # only variable 0 is forecast

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
    _train(trainer, N_CONTEXT + 2)
    seen = {}
    for name, p in trainer.model.named_parameters():
        if p.requires_grad:
            p.register_hook(lambda g, name=name: seen.__setitem__(name, True))
    trainer.train_step(_field(N_CONTEXT + 2), T0 + (N_CONTEXT + 2) * DT)
    trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    assert trainable and sorted(seen) == sorted(trainable)


def test_frozen_copy_follows_the_reconstructor_slowly():
    trainer = _trainer(ema_decay=0.9)
    for k in range(N_CONTEXT + 3):
        trainer.train_step(_field(k), T0 + k * DT)
    live = torch.cat([p.flatten() for p in trainer.model.reconstructor.parameters()])
    frozen = torch.cat([p.flatten() for p in trainer.model.frozen.module.parameters()])
    assert int(trainer.model.frozen.n_averaged) == 3
    assert not torch.equal(live, frozen)
    assert (live - frozen).abs().max() < 0.1


def _step_latent(k):
    """Latent of sample step k with 2 variables: variable 0 holds k, variable 1 holds -k."""
    return {0: torch.tensor([k, -k], dtype=torch.float32).reshape(1, 2, 1, 1, 1, 1)}


def test_replay_buffer_stores_each_step_once_and_draws_windows_without_replacement():
    buffer = ReplayBuffer(n_context=2, size=3, batch=2, generator=torch.Generator().manual_seed(0))
    assert not buffer.context_full and buffer.sample() == []
    for k in range(8):
        buffer.append(T0 + k * DT, _step_latent(k))
    assert [s for s, _ in buffer.steps] == [T0 + k * DT for k in range(3, 8)]  # n_context + size steps
    context, seconds = buffer.context()
    assert context[0][0, :, :, 0, 0, 0].tolist() == [[6, 7], [-6, -7]] and seconds == [T0 + 6 * DT, T0 + 7 * DT]
    for _ in range(20):
        drawn = buffer.sample()
        targets = [int(w.target[0].item()) for w in drawn]
        assert len(targets) == len(set(targets)) == 2 and set(targets) <= {5, 6, 7}
        for w, k in zip(drawn, targets):
            assert w.target[0].shape[1] == 1  # variable 0 only
            assert w.context[0][0, :, :, 0, 0, 0].tolist() == [[k - 2, k - 1], [2 - k, 1 - k]]
            assert w.seconds == [T0 + j * DT for j in (k - 2, k - 1, k)]
    small = ReplayBuffer(n_context=2, size=3, batch=2, generator=torch.Generator().manual_seed(0))
    for k in range(3):
        small.append(T0 + k * DT, _step_latent(k))
    assert [int(w.target[0].item()) for w in small.sample()] == [2]


def test_replay_off_draws_no_random_numbers():
    rng = torch.Generator().manual_seed(0)
    state = rng.get_state()
    buffer = ReplayBuffer(n_context=2, size=0, batch=10, generator=rng)
    for k in range(5):
        buffer.append(T0 + k * DT, _step_latent(k))
    assert len(buffer.steps) == 2 and buffer.context_full
    assert buffer.sample() == [] and torch.equal(rng.get_state(), state)


def test_batched_forecast_matches_the_single_window_forecast():
    trainer = _trainer()
    _train(trainer, N_CONTEXT)
    model = trainer.model
    context, seconds = trainer.replay_buffer.context()
    times = seconds + [T0 + N_CONTEXT * DT]
    single = model.forecast(context, [times])
    other = {z: torch.randn_like(h) for z, h in context.items()}
    batch = model.forecast({z: torch.cat([context[z], other[z], other[z]]) for z in context},
                           [times, [s + 3600 for s in times], [s + 7200 for s in times]])
    for z in single:
        assert batch[z].shape[0] == 3
        assert torch.allclose(batch[z][:1], single[z], atol=1e-5), z


def test_replay_starts_with_the_second_update():
    """The first update has an empty buffer, so it equals training without
    replay; every update stores its window."""
    off, on = _trainer(), _trainer(replay_size=4, replay_batch=2)
    for k in range(N_CONTEXT + 2):
        s_off, s_on = off.train_step(_field(k), T0 + k * DT), on.train_step(_field(k), T0 + k * DT)
        if k == N_CONTEXT:
            assert s_on == s_off
    assert s_on != s_off
    assert len(on.replay_buffer.steps) == N_CONTEXT + 2 and len(off.replay_buffer.steps) == N_CONTEXT


def test_forecaster_batch_is_the_new_window_plus_the_drawn_ones():
    """With a draw as large as the buffer, the batch of the 3rd update is
    exactly the 3 windows in the replay buffer afterwards, so its persistence reference is
    their mean."""
    trainer = _trainer(replay_size=10, replay_batch=10)
    stats = _train(trainer, N_CONTEXT + 3)
    buffer = trainer.replay_buffer
    windows = [buffer.window(end) for end in range(N_CONTEXT, len(buffer.steps))]
    assert len(windows) == 3
    for z, value in stats["latent_persist"].items():
        expected = sum(float((w.target[z] - w.context[z][:, :1, -1:]).square().mean()) for w in windows) / 3
        assert value == pytest.approx(expected, rel=1e-5), z


def test_stored_window_holds_the_context_target_latent_and_times():
    trainer = _trainer(replay_size=4)
    _train(trainer, N_CONTEXT)
    context, _ = trainer.replay_buffer.context()
    trainer.train_step(_field(N_CONTEXT), T0 + N_CONTEXT * DT)
    assert len(trainer.replay_buffer.steps) == N_CONTEXT + 1
    window = trainer.replay_buffer.window(N_CONTEXT)
    assert window.seconds == [T0 + k * DT for k in range(N_CONTEXT + 1)]
    assert all(torch.equal(window.context[z], context[z]) for z in context)
    _, cached = trainer.replay_buffer.steps[-1]  # c̄_t, added after the update
    assert all(torch.equal(window.target[z], cached[z][:, :1]) for z in cached)  # variable 0


def test_forecast_depends_on_the_forcing_variables():
    """The forecaster attends across variables, so variable 0's forecast sees
    the forcings' latents. The gates start near zero, so the dependence is tiny
    but present."""
    trainer = _trainer()
    _train(trainer, N_CONTEXT)
    context, seconds = trainer.replay_buffer.context()
    context = {z: c.clone().requires_grad_() for z, c in context.items()}
    forecast = trainer.model.forecast(context, [seconds + [T0 + N_CONTEXT * DT]])
    sum(f.sum() for f in forecast.values()).backward()
    assert all(c.grad[:, 1:].abs().sum() > 0 for c in context.values())


def test_forecast_is_decoded_with_the_forcing_latents_of_the_latest_state():
    """Forcings at the forecast time are unknown: the decoded forecast latent
    is variable 0 from the forecaster plus the forcing latents of c̄_latest."""
    trainer = _trainer()
    _train(trainer, N_CONTEXT)
    _, latest = trainer.replay_buffer.steps[-1]
    decoded, decode = [], trainer.model.decode_frozen
    trainer.model.decode_frozen = lambda latent, seconds: decoded.append(latent) or decode(latent, seconds)
    increment = trainer.predict_increment(T0 + N_CONTEXT * DT)
    forecast, last = decoded
    assert increment.shape == (N_FINE, NLEV)
    assert all(last[z] is latest[z] for z in latest)
    assert all(torch.equal(forecast[z][:, 1:], latest[z][:, 1:]) for z in latest)
    assert all(forecast[z].shape == latest[z].shape for z in latest)
