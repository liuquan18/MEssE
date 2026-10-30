"""Tests for fieldspace_RF_online.py on a small synthetic nested patch.

Levels 0, 2, 3, 4 (backbone 0, level 1 has a grid layer but no data), latent
level 2: two compression stages. Two forcings unless a test says otherwise:
column 0 conditions both models, column 1 (like the insolation) only the
forecaster. CPU only, no comin, no MPI.
"""

import pytest
import torch

from fieldspace_RF_online import ReplayBuffer, OnlineReconstructorForecaster, Window, autoencoder_configs, _BlockConfigs
from icon_nested_mgrid import patch_mgrid
from test_icon_nested_mgrid import synthetic_global_mgrid

LEVELS, LATENT_LEVEL, N_CONTEXT, NLEV = [0, 2, 3, 4], 2, 3, 1
N_FORCINGS, RECONSTRUCTOR_FORCINGS = 2, [0]
T0, DT = 1.78e9, 600.0
N_BACKBONE = 10
N_FINE = N_BACKBONE * 4**4


def _trainer(n_forcings=N_FORCINGS, **kwargs):
    torch.manual_seed(0)
    mgrids = patch_mgrid(synthetic_global_mgrid(n_levels=5), torch.arange(N_BACKBONE))
    config = dict(n_context=N_CONTEXT, device=torch.device("cpu"), att_dim=16, n_head_channels=4,
                  hidden_dim=16, embed_dim=8, n_blocks=1, n_forecaster_blocks=1, n_forcings=n_forcings,
                  reconstructor_forcings=RECONSTRUCTOR_FORCINGS[:n_forcings])
    config.update(kwargs)
    return OnlineReconstructorForecaster(mgrids, LEVELS, LATENT_LEVEL, NLEV, **config)


def _field(k):
    torch.manual_seed(100 + k)
    return torch.randn(N_FINE, NLEV)


def _forcings(k, n_forcings=N_FORCINGS):
    torch.manual_seed(200 + k)
    return torch.randn(N_FINE, n_forcings)


def _train(trainer, n_steps, n_forcings=N_FORCINGS):
    """``n_steps`` sample steps; returns the stats of the last one."""
    for k in range(n_steps):
        stats = trainer.train_step(_field(k), _forcings(k, n_forcings), T0 + k * DT)
    return stats


def _predict(trainer, k, n_forcings=N_FORCINGS):
    """The increment to sample step k, with the forcings of step k."""
    return trainer.predict_increment(T0 + k * DT, _forcings(k, n_forcings))


def test_levels_are_backbone_mean_plus_zero_mean_residuals_and_invert_exactly():
    model = _trainer().model
    x = _field(0)
    x_copy = x.clone()
    levels = model.to_levels(x)
    assert torch.equal(x, x_copy)  # input untouched (encode_zooms works in place)
    assert sorted(levels) == [0, 2, 3, 4]
    assert torch.allclose(levels[0][0, 0, 0, :, 0, 0], x.reshape(N_BACKBONE, -1).mean(dim=1), atol=1e-6)
    for parent, zoom in [(0, 2), (2, 3), (3, 4)]:
        residual = levels[zoom].reshape(-1, 4 ** (zoom - parent))
        assert torch.allclose(residual.mean(dim=1), torch.zeros(1), atol=1e-5)
    assert torch.allclose(model.to_field(levels), x, atol=1e-5)


def test_forcing_levels_are_the_means_over_each_zooms_cells():
    model = _trainer().model
    h = _forcings(0)
    levels = model.forcing_levels(h)
    assert sorted(levels) == [0, 2, 3, 4]
    for zoom, f in levels.items():
        assert f.shape == (1, 1, N_BACKBONE * 4**zoom, N_FORCINGS)
        expected = h.reshape(N_BACKBONE * 4**zoom, -1, N_FORCINGS).mean(dim=1)
        assert torch.allclose(f[0, 0], expected, atol=1e-6), zoom
    assert model.forcing_levels(torch.zeros(N_FINE, 0)) == {}


@pytest.mark.parametrize("latent_zoom, latent", [(2, {0: 1, 2: 4}), (3, {0: 1, 2: 1, 3: 4})])
def test_latent_levels_and_channels(latent_zoom, latent):
    *_, latent_features = autoencoder_configs([0, 2, 3, 4], latent_zoom, 4, 1, _BlockConfigs(16, 4, 16, 8, 0))
    assert latent_features == latent


def test_context_fills_before_training_and_forecasting():
    trainer = _trainer()
    for k in range(N_CONTEXT):
        assert trainer.train_step(_field(k), _forcings(k), T0 + k * DT) == {}
    assert _predict(trainer, N_CONTEXT) is not None
    stats = trainer.train_step(_field(N_CONTEXT), _forcings(N_CONTEXT), T0 + N_CONTEXT * DT)
    assert set(stats) == {"recon", "state_ms", "latent_mse", "latent_persist"}
    assert sorted(stats["recon"]) == sorted(stats["state_ms"]) == [0, 2, 3, 4]
    assert sorted(stats["latent_mse"]) == sorted(stats["latent_persist"]) == [0, 2]  # latent levels


@pytest.mark.parametrize("n_forcings", [0, N_FORCINGS])
def test_untrained_forecast_is_persistence(n_forcings):
    trainer = _trainer(n_forcings)
    _train(trainer, N_CONTEXT, n_forcings)
    increment = _predict(trainer, N_CONTEXT, n_forcings)
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


def _forward(trainer, k=N_CONTEXT):
    """The training pass at sample step k on the live window, with the
    forcing levels made leaves that collect gradients."""
    model = trainer.model
    context, context_forcings, seconds = trainer.replay_buffer.context()
    forcings = {z: f.clone().requires_grad_() for z, f in model.forcing_levels(_forcings(k)).items()}
    window_forcings = {z: torch.cat([f, forcings[z].detach()], dim=1).requires_grad_()
                       for z, f in context_forcings.items()}
    window = Window(context, {}, window_forcings, [seconds + [T0 + k * DT]])  # a batch of one; no target needed
    recon, forecast = model(model.to_levels(_field(k)), T0 + k * DT, forcings, window)
    return recon, forecast, forcings, window_forcings


def test_models_are_trained_independently():
    """The forecast loss never reaches the reconstructor, the reconstruction
    loss never reaches the forecaster, and the frozen copy gets no gradient."""
    trainer = _trainer()
    _train(trainer, N_CONTEXT)
    model = trainer.model
    recon, forecast, _, _ = _forward(trainer)
    assert all(r.shape[1] == 1 for r in recon.values()) and all(f.shape[1] == 1 for f in forecast.values())

    sum(f.square().sum() for f in forecast.values()).backward(retain_graph=True)
    assert all(p.grad is None for p in model.reconstructor.parameters())
    assert all(p.grad is None for p in model.frozen.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.forecaster.parameters())

    model.zero_grad(set_to_none=True)
    sum(r.square().sum() for r in recon.values()).backward()
    assert all(p.grad is None for p in model.forecaster.parameters())


def test_reconstructor_sees_only_its_forcings_at_every_zoom():
    """The gates start near zero, so the dependence on the forcings is tiny
    but present: on column 0 at every field zoom, never on column 1."""
    trainer = _trainer()
    _train(trainer, N_CONTEXT)
    recon, _, forcings, _ = _forward(trainer)
    sum(r.sum() for r in recon.values()).backward()
    for zoom, f in forcings.items():
        if zoom in (2, 3, 4):  # the finest zoom of some reconstructor block
            assert f.grad[..., 0].abs().sum() > 0, zoom
        assert f.grad is None or torch.equal(f.grad[..., 1], torch.zeros_like(f.grad[..., 1])), zoom


def test_forecaster_sees_every_forcing_of_every_slot_on_its_latent_zoom():
    trainer = _trainer()
    _train(trainer, N_CONTEXT)
    _, forecast, _, window_forcings = _forward(trainer)
    assert sorted(window_forcings) == [2]
    sum(f.sum() for f in forecast.values()).backward()
    grad = window_forcings[2].grad
    assert grad.shape[1] == N_CONTEXT + 1  # context steps, then the forecast slot
    assert all(grad[:, slot, :, column].abs().sum() > 0 for slot in range(N_CONTEXT + 1) for column in range(2))


@pytest.mark.parametrize("replay", [{}, {"replay_size": 4}])
def test_every_trainable_parameter_gets_a_gradient_in_one_pass(replay):
    """DDP (find_unused_parameters=False) needs this, with and without replay."""
    trainer = _trainer(**replay)
    _train(trainer, N_CONTEXT + 2)
    seen = {}
    for name, p in trainer.model.named_parameters():
        if p.requires_grad:
            p.register_hook(lambda g, name=name: seen.__setitem__(name, True))
    k = N_CONTEXT + 2
    trainer.train_step(_field(k), _forcings(k), T0 + k * DT)
    trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    assert any("CellForcingEmbedder" in n for n in trainable)
    assert trainable and sorted(seen) == sorted(trainable)


def test_frozen_copy_follows_the_reconstructor_slowly():
    trainer = _trainer(ema_decay=0.9)
    _train(trainer, N_CONTEXT + 3)
    live = torch.cat([p.flatten() for p in trainer.model.reconstructor.parameters()])
    frozen = torch.cat([p.flatten() for p in trainer.model.frozen.module.parameters()])
    assert int(trainer.model.frozen.n_averaged) == 3
    assert not torch.equal(live, frozen)
    assert (live - frozen).abs().max() < 0.1


def _step(k):
    """Latent and forcings of sample step k: the latent holds k, the forcings -k."""
    return ({0: torch.full((1, 1, 1, 1, 1, 1), float(k))},
            {0: torch.full((1, 1, 1, 1), -float(k))})


def test_replay_buffer_stores_each_step_once_and_draws_windows_without_replacement():
    buffer = ReplayBuffer(n_context=2, size=3, batch=2, generator=torch.Generator().manual_seed(0))
    assert not buffer.context_full and buffer.sample() == []
    for k in range(8):
        buffer.append(T0 + k * DT, *_step(k))
    assert [s for s, _, _ in buffer.steps] == [T0 + k * DT for k in range(3, 8)]  # n_context + size steps
    context, forcings, seconds = buffer.context()
    assert context[0].flatten().tolist() == [6, 7] and seconds == [T0 + 6 * DT, T0 + 7 * DT]
    assert forcings[0].flatten().tolist() == [-6, -7]
    for _ in range(20):
        drawn = buffer.sample()
        targets = [int(w.target[0].item()) for w in drawn]
        assert len(targets) == len(set(targets)) == 2 and set(targets) <= {5, 6, 7}
        for w, k in zip(drawn, targets):
            assert w.context[0].flatten().tolist() == [k - 2, k - 1]
            assert w.forcings[0].flatten().tolist() == [2 - k, 1 - k, -k]  # context steps, then the target's
            assert w.seconds == [T0 + j * DT for j in (k - 2, k - 1, k)]
    small = ReplayBuffer(n_context=2, size=3, batch=2, generator=torch.Generator().manual_seed(0))
    for k in range(3):
        small.append(T0 + k * DT, *_step(k))
    assert [int(w.target[0].item()) for w in small.sample()] == [2]


def test_replay_buffer_never_draws_windows_touching_the_warmup_steps():
    """With warmup=3, steps 0-2 never appear in a drawn window, also after
    the deque has pushed old steps out."""
    buffer = ReplayBuffer(n_context=2, size=6, batch=10, generator=torch.Generator().manual_seed(0), warmup=3)

    def drawn_targets():
        return sorted(int(w.target[0].item()) for w in buffer.sample())

    for k in range(5):
        buffer.append(T0 + k * DT, *_step(k))
    assert drawn_targets() == []  # every stored window still has warm-up steps in its context
    buffer.append(T0 + 5 * DT, *_step(5))
    assert drawn_targets() == [5]  # context 3, 4
    for k in range(6, 12):
        buffer.append(T0 + k * DT, *_step(k))
    assert [int(lv[0].item()) for _, lv, _ in buffer.steps] == list(range(4, 12))
    assert drawn_targets() == list(range(6, 12))  # target 5 lost its context step 3


def test_replay_off_draws_no_random_numbers():
    rng = torch.Generator().manual_seed(0)
    state = rng.get_state()
    buffer = ReplayBuffer(n_context=2, size=0, batch=10, generator=rng)
    for k in range(5):
        buffer.append(T0 + k * DT, *_step(k))
    assert len(buffer.steps) == 2 and buffer.context_full
    assert buffer.sample() == [] and torch.equal(rng.get_state(), state)


def test_batched_forecast_matches_the_single_window_forecast():
    trainer = _trainer()
    _train(trainer, N_CONTEXT)
    model = trainer.model
    context, forcings, seconds = trainer.replay_buffer.context()
    forcings = {z: torch.cat([f, f[:, -1:]], dim=1) for z, f in forcings.items()}
    times = seconds + [T0 + N_CONTEXT * DT]
    single = model.forecast(context, forcings, [times])
    other = {z: torch.randn_like(c) for z, c in context.items()}
    other_forcings = {z: torch.randn_like(f) for z, f in forcings.items()}
    batch = model.forecast({z: torch.cat([context[z], other[z], other[z]]) for z in context},
                           {z: torch.cat([forcings[z], other_forcings[z], forcings[z]]) for z in forcings},
                           [times, [s + 3600 for s in times], [s + 7200 for s in times]])
    for z in single:
        assert batch[z].shape[0] == 3
        assert torch.allclose(batch[z][:1], single[z], atol=1e-5), z


def test_replay_starts_with_the_second_update():
    """The first update has an empty buffer, so it equals training without
    replay; every update stores its window."""
    off, on = _trainer(), _trainer(replay_size=4, replay_batch=2)
    for k in range(N_CONTEXT + 2):
        s_off = off.train_step(_field(k), _forcings(k), T0 + k * DT)
        s_on = on.train_step(_field(k), _forcings(k), T0 + k * DT)
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
        expected = sum(float((w.target[z] - w.context[z][:, :, -1:]).square().mean()) for w in windows) / 3
        assert value == pytest.approx(expected, rel=1e-5), z


def test_stored_window_holds_the_context_target_forcings_and_times():
    trainer = _trainer(replay_size=4)
    _train(trainer, N_CONTEXT)
    context, _, _ = trainer.replay_buffer.context()
    trainer.train_step(_field(N_CONTEXT), _forcings(N_CONTEXT), T0 + N_CONTEXT * DT)
    assert len(trainer.replay_buffer.steps) == N_CONTEXT + 1
    window = trainer.replay_buffer.window(N_CONTEXT)
    assert window.seconds == [T0 + k * DT for k in range(N_CONTEXT + 1)]
    assert all(torch.equal(window.context[z], context[z]) for z in context)
    _, cached, _ = trainer.replay_buffer.steps[-1]  # c̄_t, added after the update
    assert all(torch.equal(window.target[z], cached[z]) for z in cached)
    expected = [trainer.model.forcing_levels(_forcings(k))[2] for k in range(N_CONTEXT + 1)]
    assert torch.equal(window.forcings[2], torch.cat(expected, dim=1))  # forecaster forcings, on its latent zoom


def test_forecast_slot_and_both_decodes_use_the_given_forcings_and_the_latest_time():
    trainer = _trainer()
    _train(trainer, N_CONTEXT)
    latest_seconds, latest, _ = trainer.replay_buffer.steps[-1]
    model = trainer.model
    decoded, decode = [], model.decode_frozen
    model.decode_frozen = lambda latent, seconds, forcings: decoded.append((seconds, forcings)) or decode(latent, seconds, forcings)
    slots, forecast = [], model.forecast
    model.forecast = lambda context, forcings, times: slots.append(forcings) or forecast(context, forcings, times)
    h = _forcings(N_CONTEXT)
    increment = trainer.predict_increment(T0 + N_CONTEXT * DT, h)
    assert increment.shape == (N_FINE, NLEV)
    (s_forecast, f_forecast), (s_last, f_last) = decoded
    assert s_forecast == s_last == latest_seconds and f_forecast is f_last
    assert all(torch.equal(f_forecast[z], f) for z, f in model.forcing_levels(h).items())
    assert torch.equal(slots[0][2][:, -1:], model.forcing_levels(h)[2])  # the forecast slot


def test_persistence_forecast_gives_exactly_zero_increment():
    """If the forecaster returns the latest latent, the two decodes see the
    same input with the same forcings at the same time, so nothing the decoder
    does with them can leak into the increment."""
    trainer = _trainer()
    _train(trainer, N_CONTEXT)
    _, latest, _ = trainer.replay_buffer.steps[-1]
    trainer.model.forecast = lambda context, forcings, times: dict(latest)
    assert torch.equal(_predict(trainer, N_CONTEXT), torch.zeros(N_FINE, NLEV))


def test_reconstructor_without_time_ignores_the_clock_but_the_forecaster_does_not():
    trainer = _trainer(reconstructor_time=False)
    model = trainer.model
    levels, forcings = model.to_levels(_field(0)), model.forcing_levels(_forcings(0))
    with torch.no_grad():
        noon, midnight = model.encode_frozen(levels, T0, forcings), model.encode_frozen(levels, T0 + 43200, forcings)
        assert all(torch.equal(noon[z], midnight[z]) for z in noon)
        decoded = [model.decode_frozen(noon, s, forcings) for s in (T0, T0 + 43200)]
        assert all(torch.equal(decoded[0][z], decoded[1][z]) for z in decoded[0])
    _train(trainer, N_CONTEXT)
    seen, forecaster = [], model.forecaster.forward
    model.forecaster.forward = lambda context, emb: seen.append(emb["TimeEmbedder"][0].clone()) or forecaster(context, emb)
    _predict(trainer, N_CONTEXT)
    expected = torch.tensor([[(T0 + k * DT) / 86400 for k in range(N_CONTEXT + 1)]])
    assert torch.allclose(seen[0], expected)  # context times then the forecast time, in days
