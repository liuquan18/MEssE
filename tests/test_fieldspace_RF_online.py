"""Tests for fieldspace_RF_online.py on a small synthetic nested patch.

Levels 0, 2, 3, 4 (backbone 0, level 1 has a grid layer but no data), latent
level 2: two compression stages. CPU only, no comin, no MPI.
"""

import pytest
import torch

from fieldspace_RF_online import OnlineReconstructorForecaster, autoencoder_configs, _Blocks
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


def test_history_fills_before_training_and_forecasting():
    trainer = _trainer()
    for k in range(N_HISTORY):
        assert trainer.train_step(_field(k), T0 + k * DT) == {}
    assert trainer.predict_increment(T0 + N_HISTORY * DT) is not None
    stats = trainer.train_step(_field(N_HISTORY), T0 + N_HISTORY * DT)
    assert set(stats) == {"recon", "incre", "oracle", "grad_norm"}
    assert sorted(stats["recon"]) == sorted(stats["incre"]) == sorted(stats["oracle"]) == [0, 2, 3, 4]


def test_untrained_forecast_is_persistence():
    trainer = _trainer()
    for k in range(N_HISTORY):
        trainer.train_step(_field(k), T0 + k * DT)
    increment = trainer.predict_increment(T0 + N_HISTORY * DT)
    assert increment.shape == (N_FINE, NLEV)
    assert increment.abs().max() < 1e-4


def test_first_increment_loss_of_persistence_is_one_per_level():
    """Each level's loss is divided by the running mean square of its target,
    so on the first update a zero increment scores exactly 1 per level."""
    trainer = _trainer()
    for k in range(N_HISTORY + 1):
        stats = trainer.train_step(_field(k), T0 + k * DT)
    for zoom, value in stats["incre"].items():
        assert value == pytest.approx(1.0, abs=1e-3), zoom


def test_backbone_is_not_compressed_so_its_oracle_is_exact():
    trainer = _trainer()
    for k in range(N_HISTORY + 1):
        stats = trainer.train_step(_field(k), T0 + k * DT)
    assert stats["oracle"][0] < 1e-6


def test_models_are_trained_independently():
    """The forecast loss never reaches the reconstructor, the reconstruction
    loss never reaches the forecaster, and the frozen copy gets no gradient."""
    trainer = _trainer()
    for k in range(N_HISTORY):
        trainer.train_step(_field(k), T0 + k * DT)
    model = trainer.model
    history, seconds = trainer._history()
    levels = model.to_levels(_field(9))
    recon, forecast = model(levels, T0 + N_HISTORY * DT, history, seconds)

    sum(f.square().sum() for f in forecast.values()).backward(retain_graph=True)
    assert all(p.grad is None for p in model.reconstructor.parameters())
    assert all(p.grad is None for p in model.frozen.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.forecaster.parameters())

    model.zero_grad(set_to_none=True)
    sum(r.square().sum() for r in recon.values()).backward()
    assert all(p.grad is None for p in model.forecaster.parameters())


def test_every_trainable_parameter_gets_a_gradient_in_one_pass():
    """DDP (find_unused_parameters=False) needs this."""
    trainer = _trainer()
    for k in range(N_HISTORY):
        trainer.train_step(_field(k), T0 + k * DT)
    seen = {}
    for name, p in trainer.model.named_parameters():
        if p.requires_grad:
            p.register_hook(lambda g, name=name: seen.__setitem__(name, True))
    trainer.train_step(_field(N_HISTORY), T0 + N_HISTORY * DT)
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
