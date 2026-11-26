"""Tests for fieldspace_AE_online.py on a small synthetic nested patch.

Levels 0, 2, 3, 4 (backbone 0, level 1 has a grid layer but no data), latent
level 2: two compression stages. CPU only, no comin, no MPI.
"""

import math

import pytest
import torch

from fieldspace_AE_online import OnlineFieldSpaceAE
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
                  hidden_dim=16, time_embed_dim=8, n_blocks=1, n_processor_blocks=1)
    config.update(kwargs)
    return OnlineFieldSpaceAE(mgrids, LEVELS, LATENT_LEVEL, NLEV, **config)


def _field(k):
    torch.manual_seed(100 + k)
    return torch.randn(N_FINE, NLEV)


def _filled(trainer):
    for k in range(N_HISTORY):
        assert trainer.train_step(_field(k), T0 + k * DT) == {}
    return trainer


def test_levels_invert_exactly():
    model = _trainer().model
    x = _field(0)
    levels = model.to_levels(x)
    assert sorted(levels) == [0, 2, 3, 4]
    assert torch.allclose(model.to_field(levels), x, atol=1e-5)


def test_latent_is_backbone_plus_compressed_level():
    assert _trainer().model.latent_features == {0: 1, 2: 4}


def test_history_fills_before_training_and_forecasting():
    trainer = _filled(_trainer())
    assert trainer.predict_increment(T0 + N_HISTORY * DT) is not None
    stats = trainer.train_step(_field(N_HISTORY), T0 + N_HISTORY * DT)
    assert set(stats) == {"loss", "recon", "incre", "oracle", "persist_mse", "state_ms", "grad_norm"}
    assert sorted(stats["recon"]) == sorted(stats["incre"]) == sorted(stats["oracle"]) == [0, 2, 3, 4]
    assert len(trainer.history) == N_HISTORY


def test_untrained_forecast_is_persistence():
    increment = _filled(_trainer()).predict_increment(T0 + N_HISTORY * DT)
    assert increment.shape == (N_FINE, NLEV)
    assert increment.abs().max() < 1e-4


def test_untrained_increment_loss_is_the_persistence_mse():
    """An untrained model forecasts zero change, so its increment loss per
    level is the logged persistence reference."""
    stats = _filled(_trainer()).train_step(_field(N_HISTORY), T0 + N_HISTORY * DT)
    for zoom, value in stats["incre"].items():
        assert value == pytest.approx(stats["persist_mse"][zoom], rel=1e-3), zoom


def test_logged_parts_are_the_trained_loss():
    """The log carries the unscaled per-level MSEs that are trained on: their
    sum is the trained total."""
    stats = _filled(_trainer()).train_step(_field(N_HISTORY), T0 + N_HISTORY * DT)
    assert stats["loss"] == pytest.approx(sum(stats["recon"].values()) + sum(stats["incre"].values()), rel=1e-5)


def test_backbone_is_not_compressed_so_its_oracle_is_exact():
    stats = _filled(_trainer()).train_step(_field(N_HISTORY), T0 + N_HISTORY * DT)
    assert stats["recon"][0] < 1e-6


def test_increment_loss_trains_processor_and_decoder_not_encoder():
    """One stage: the forecast is decoded by the live decoder, so the
    increment loss reaches the decoder too; the cached history is detached,
    so it never reaches the encoder. A few updates first open the processor's
    gates: at init the forecast is exactly persistence, so D(ĉ) - D(c) has no
    gradient at all."""
    trainer = _trainer(lr=1e-2)
    for k in range(N_HISTORY + 3):
        trainer.train_step(_field(k), T0 + k * DT)
    model = trainer.model
    model.zero_grad(set_to_none=True)
    history, seconds = trainer._history()
    levels = model.to_levels(_field(9))
    _, recon, forecast, prev = model(levels, T0 + 9 * DT, history, seconds, seconds[-1])

    sum((forecast[z] - prev[z] - 1).square().sum() for z in forecast).backward()
    grads = {n: p.grad is not None and bool(p.grad.abs().sum() > 0) for n, p in model.named_parameters()}
    assert any(v for n, v in grads.items() if n.startswith("processor."))
    assert any(v for n, v in grads.items() if n.startswith("autoencoder.decoder"))
    assert not any(v for n, v in grads.items() if n.startswith("autoencoder.encoder"))


def test_every_trainable_parameter_gets_a_gradient_in_one_pass():
    """DDP (find_unused_parameters=False) needs this."""
    trainer = _filled(_trainer())
    seen = {}
    for name, p in trainer.model.named_parameters():
        if p.requires_grad:
            p.register_hook(lambda g, name=name: seen.__setitem__(name, True))
    trainer.train_step(_field(N_HISTORY), T0 + N_HISTORY * DT)
    trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    assert trainable and sorted(seen) == sorted(trainable)


def test_training_beats_persistence_on_an_oscillation():
    """x_k = base + sin(2 pi k / 8) d: bounded, and its change is predictable
    from the history, so training must move below persistence. Slowly: with
    the plain loss the reconstruction dominates the gradient first."""
    g = torch.Generator().manual_seed(1)
    base, d = torch.randn(N_FINE, NLEV, generator=g), torch.randn(N_FINE, NLEV, generator=g)
    trainer = _trainer(lr=1e-2)
    relative = []
    for k in range(60):
        stats = trainer.train_step(base + math.sin(2 * math.pi * k / 8) * d, T0 + k * DT)
        if stats:
            relative.append(sum(stats["incre"].values()) / sum(stats["persist_mse"].values()))
    assert relative[0] == pytest.approx(1.0, abs=1e-3)
    assert sum(relative[-8:]) / 8 < 0.98
