"""Unit tests for gnn_online.py: the GNN model itself and OnlineGNNTrainer.

(RunningMeanStd/RolloutBuffer — the generic online-training bookkeeping
gnn_plugin.py also uses — live in MEssE.utils.icon_online_helper and are
tested in test_icon_online_helper.py.)

All of this runs on CPU with plain synthetic tensors — no comin, no MPI, no
GPU, no live ICON simulation required. This is deliberately the fast
feedback loop for changes to the GNN training logic; a real correctness
check against ICON still requires an actual `run_icon_gpu.sh` submission
(slow, GPU-queue-gated — see CLAUDE.md), which these tests are meant to
reduce how often you need.
"""

import math

import pytest
import torch

from gnn_online import GNNModel, GNNSnapshot, OnlineGNNTrainer
from graph_utils import LocalGraph


# ----------------------------------------------------------------------------
# GNNModel
# ----------------------------------------------------------------------------


def _tiny_graph(n_nodes: int = 4, owned=(True, True, False, False)) -> LocalGraph:
    # A small symmetric ring 0-1-2-3-0 plus self-loops, on CPU.
    src = [0, 1, 1, 2, 2, 3, 3, 0] + list(range(n_nodes))
    dst = [1, 0, 2, 1, 3, 2, 0, 3] + list(range(n_nodes))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], 3)
    return LocalGraph(
        n_nodes=n_nodes,
        n_valid=n_nodes,
        edge_index=edge_index,
        owned_mask=torch.tensor(owned, dtype=torch.bool),
        clon=torch.zeros(n_nodes),
        clat=torch.zeros(n_nodes),
        edge_attr=edge_attr,
    )


def test_gnn_model_forward_preserves_shape():
    nlev = 5
    graph = _tiny_graph()
    model = GNNModel(nlev=nlev, hidden_dim=8, n_processor_layers=2)

    x = torch.randn(graph.n_nodes, nlev)
    node_static = torch.stack([graph.clon, graph.clat], dim=-1)
    out = model(x, node_static, graph.edge_attr, graph.edge_index)

    assert out.shape == x.shape


# ----------------------------------------------------------------------------
# OnlineGNNTrainer
# ----------------------------------------------------------------------------


def _tiny_trainer(nlev: int = 3) -> OnlineGNNTrainer:
    graph = _tiny_graph()
    return OnlineGNNTrainer(
        nlev=nlev,
        graph=graph,
        hidden_dim=8,
        n_processor_layers=1,
        grad_clip=1.0,
        use_ddp=False,
        device=torch.device("cpu"),
        log_fn=lambda msg: None,
        rank=0,
    )


def test_masked_mse_ignores_halo_nodes():
    trainer = _tiny_trainer()
    n_nodes, nlev = trainer.graph.n_nodes, trainer.nlev

    pred = torch.zeros(n_nodes, nlev)
    target = torch.zeros(n_nodes, nlev)
    # Introduce error only on halo (non-owned) nodes 2 and 3.
    target[2:] = 100.0

    loss = trainer._masked_mse(pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

    # Now introduce the same error on an owned node -> loss must move.
    target2 = target.clone()
    target2[0] = 1.0
    loss2 = trainer._masked_mse(pred, target2)
    assert loss2.item() > 0.0


def test_train_rollout_step_skips_on_nan_source():
    trainer = _tiny_trainer()
    n_nodes, nlev = trainer.graph.n_nodes, trainer.nlev

    source = GNNSnapshot(x=torch.full((n_nodes, nlev), float("nan")), unix_seconds=0.0)
    target = GNNSnapshot(x=torch.zeros(n_nodes, nlev), unix_seconds=1.0)

    result = trainer.train_rollout_step(source, target, n_steps=2)
    assert result["skipped"] is True
    assert math.isnan(result["loss"])


def test_train_rollout_step_skips_on_nan_target():
    trainer = _tiny_trainer()
    n_nodes, nlev = trainer.graph.n_nodes, trainer.nlev

    source = GNNSnapshot(x=torch.zeros(n_nodes, nlev), unix_seconds=0.0)
    target = GNNSnapshot(x=torch.full((n_nodes, nlev), float("nan")), unix_seconds=1.0)

    result = trainer.train_rollout_step(source, target, n_steps=2)
    assert result["skipped"] is True
    assert math.isnan(result["loss"])


def test_train_rollout_step_runs_and_reports_finite_metrics():
    torch.manual_seed(0)
    trainer = _tiny_trainer()
    n_nodes, nlev = trainer.graph.n_nodes, trainer.nlev

    source = trainer.prepare_snapshot(torch.randn(n_nodes, nlev) * 0.1, unix_seconds=0.0)
    target = trainer.prepare_snapshot(torch.randn(n_nodes, nlev) * 0.1, unix_seconds=2.0)

    result = trainer.train_rollout_step(source, target, n_steps=2)
    assert result["skipped"] is False
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["grad_norm"])
    assert set(result["loss_dict"].keys()) == {"train/MSE"}
    assert result["loss_dict"]["train/MSE"] == pytest.approx(result["loss"])


def test_train_rollout_step_takes_exactly_n_forward_passes():
    """The loss is single-target, but the model computation must still roll
    forward autoregressively through the full n_steps gap (using its own
    predictions) before comparing to that one target — not just 1 step."""
    trainer = _tiny_trainer()
    n_nodes, nlev = trainer.graph.n_nodes, trainer.nlev

    n_forward_calls = 0
    real_forward = trainer._forward

    def counting_forward(x):
        nonlocal n_forward_calls
        n_forward_calls += 1
        return real_forward(x)

    trainer._forward = counting_forward

    source = trainer.prepare_snapshot(torch.randn(n_nodes, nlev) * 0.1, unix_seconds=0.0)
    target = trainer.prepare_snapshot(torch.randn(n_nodes, nlev) * 0.1, unix_seconds=3.0)
    trainer.train_rollout_step(source, target, n_steps=3)

    assert n_forward_calls == 3


def test_predict_preserves_shape():
    trainer = _tiny_trainer()
    n_nodes, nlev = trainer.graph.n_nodes, trainer.nlev
    snapshot = trainer.prepare_snapshot(torch.randn(n_nodes, nlev), unix_seconds=0.0)

    pred = trainer.predict(snapshot, n_steps=3)
    assert pred.shape == (n_nodes, nlev)
