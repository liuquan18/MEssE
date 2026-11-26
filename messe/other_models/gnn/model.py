"""Message-passing GNN (encoder-processor-decoder) for the ICON native grid.

Architecture follows the general GraphCast/anemoi "encoder - processor -
decoder" pattern (see https://github.com/ecmwf/anemoi), but message
passing is implemented directly in PyTorch (scatter-add via
``index_add_``) rather than depending on ``torch_geometric``/``anemoi``,
since neither is a prerequisite of this repository and the local patch
graph built in :mod:`graph` is small enough (one ICON rank's
domain-decomposed cells) that a hand-rolled interaction network is
sufficient. If ``anemoi-models``/``torch_geometric`` are available in the
runtime environment this module can be swapped for their processor
blocks without changing the plugin-facing API (`GNNSnapshot`,
`OnlineGNNTrainer`).

The model predicts, per node, the *tendency* (residual) added to the
current normalized state to obtain the next step, which empirically
trains more stably than predicting the absolute state directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from messe.other_models.gnn.graph import LocalGraph


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int = 1) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class InteractionBlock(nn.Module):
    """One round of edge-then-node message passing with residual updates."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_mlp = _mlp(3 * hidden_dim, hidden_dim, hidden_dim)
        self.node_mlp = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg_in = torch.cat([h[src], h[dst], e], dim=-1)
        e = e + self.edge_norm(self.edge_mlp(msg_in))

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, e)

        h = h + self.node_norm(self.node_mlp(torch.cat([h, agg], dim=-1)))
        return h, e


class GNNModel(nn.Module):
    """Encoder - K x InteractionBlock (processor) - Decoder."""

    def __init__(
        self,
        nlev: int,
        hidden_dim: int = 128,
        n_processor_layers: int = 6,
        n_node_static_features: int = 2,  # clon, clat
        n_edge_features: int = 3,  # dlon, dlat, great-circle distance
    ) -> None:
        super().__init__()
        self.nlev = nlev
        self.node_encoder = _mlp(nlev + n_node_static_features, hidden_dim, hidden_dim)
        self.edge_encoder = _mlp(n_edge_features, hidden_dim, hidden_dim)
        self.processor = nn.ModuleList(
            [InteractionBlock(hidden_dim) for _ in range(n_processor_layers)]
        )
        self.decoder = _mlp(hidden_dim, hidden_dim, nlev)

    def forward(
        self,
        x: torch.Tensor,  # (N, nlev) normalized node state
        node_static: torch.Tensor,  # (N, n_node_static_features)
        edge_attr: torch.Tensor,  # (E, n_edge_features)
        edge_index: torch.Tensor,  # (2, E)
    ) -> torch.Tensor:
        h = self.node_encoder(torch.cat([x, node_static], dim=-1))
        e = self.edge_encoder(edge_attr)
        for block in self.processor:
            h, e = block(h, e, edge_index)
        tendency = self.decoder(h)
        return x + tendency  # residual/tendency prediction


@dataclass
class GNNSnapshot:
    x: torch.Tensor  # (N, nlev) normalized node features, float32, on GPU
    unix_seconds: float


class OnlineGNNTrainer:
    """DDP-wrapped GNN trainer performing rollout-based online training.

    Each GPU/rank holds its own local patch graph (this rank's ICON
    domain-decomposed cells, halos included as message-passing-only
    nodes). DDP synchronizes *gradients* across ranks after each rollout,
    exactly like the UNet/HEALPix trainer — the difference is that here
    each "sample" is one rank's irregular native-grid patch instead of a
    regular HEALPix face grid.
    """

    def __init__(
        self,
        nlev: int,
        graph: LocalGraph,
        lr: float = 2e-4,
        hidden_dim: int = 128,
        n_processor_layers: int = 6,
        grad_clip: Optional[float] = 1.0,
        use_ddp: Optional[bool] = None,
        device: Optional[torch.device] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        rank: Optional[int] = None,
    ) -> None:
        self.nlev = int(nlev)
        self.device = torch.device("cuda", 0) if device is None else device
        self.grad_clip = grad_clip
        self.rank = rank
        self.log_fn = log_fn if log_fn is not None else (lambda msg: None)
        self.graph = graph

        # Static per-node features (coordinates), shared by every sample.
        self.node_static = torch.stack([graph.clon, graph.clat], dim=-1).to(self.device)
        self.edge_index = graph.edge_index.to(self.device)
        self.edge_attr = graph.edge_attr.to(self.device)
        self.owned_mask = graph.owned_mask.to(self.device)

        self.model = GNNModel(
            nlev=nlev, hidden_dim=hidden_dim, n_processor_layers=n_processor_layers
        ).to(self.device)
        self.forward_model = self._wrap_ddp(use_ddp)
        self.optimizer = torch.optim.Adam(
            self.forward_model.parameters(), lr=lr, weight_decay=0.0
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        self.log_fn(
            f"[rank={rank}] GNN initialized: nlev={nlev}, hidden_dim={hidden_dim}, "
            f"processor_layers={n_processor_layers}, nodes={graph.n_nodes}, "
            f"owned={int(self.owned_mask.sum())}, edges={self.edge_index.shape[1]}, "
            f"params={n_params:,}, device={self.device}, "
            f"ddp={isinstance(self.forward_model, DDP)}"
        )

    def _wrap_ddp(self, use_ddp: Optional[bool]) -> nn.Module:
        if use_ddp is None:
            use_ddp = dist.is_available() and dist.is_initialized()
        if not use_ddp:
            return self.model
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size <= 1:
            return self.model
        # Every rank's graph has a different number of nodes/edges (patch
        # sizes vary with the domain decomposition), but the *model*
        # parameters are shape-identical across ranks, so standard DDP
        # gradient all-reduce applies unmodified.
        return DDP(
            self.model,
            device_ids=[self.device.index or 0],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_model(x, self.node_static, self.edge_attr, self.edge_index)

    @torch.no_grad()
    def prepare_snapshot(self, x: torch.Tensor, unix_seconds: float) -> GNNSnapshot:
        x = x.to(self.device, dtype=torch.float32, non_blocking=True).detach()
        return GNNSnapshot(x=x, unix_seconds=float(unix_seconds))

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self.owned_mask.unsqueeze(-1)
        diff2 = (pred - target) ** 2 * mask
        return diff2.sum() / (mask.sum() * pred.shape[-1]).clamp(min=1)

    def train_rollout_step(
        self,
        source: GNNSnapshot,
        target: GNNSnapshot,
        n_steps: int,
    ) -> Dict[str, Any]:
        """Roll the model forward autoregressively for ``n_steps`` steps
        from ``source`` (feeding each step's own prediction back in as the
        next input — teacher forcing is *not* used), then evaluate the loss
        once against ``target``, the single ground-truth snapshot observed
        ``n_steps`` steps after ``source``. Only the final step is compared
        against real data; the intermediate ``n_steps - 1`` forward passes
        run on the model's own predictions with no supervision.
        Loss and reported metrics only consider owned (non-halo) nodes so
        that cells duplicated across neighboring ranks' halos are not
        double-counted in the (DDP-averaged) global loss.
        """
        if not torch.isfinite(source.x).all():
            self.log_fn("[trainer] NaN/Inf in source — skipping rollout step")
            return self._skip_result()
        if not torch.isfinite(target.x).all():
            self.log_fn("[trainer] NaN/Inf in target — skipping rollout step")
            return self._skip_result()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        x = source.x
        for _ in range(n_steps):
            x = self._forward(x)
        loss = self._masked_mse(x, target.x)

        if not torch.isfinite(loss):
            self.log_fn(
                f"[trainer] Non-finite rollout loss ({loss.item()}) — model may be corrupted"
            )
            return {
                "loss": loss.item(),
                "loss_dict": {},
                "skipped": True,
                "needs_rollback": True,
                "grad_norm": 0.0,
            }

        loss.backward()

        max_norm = self.grad_clip if self.grad_clip is not None else float("inf")
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm).item()

        if not math.isfinite(grad_norm):
            self.log_fn(
                f"[trainer] Non-finite grad_norm ({grad_norm}) — skipping optimizer step"
            )
            self.optimizer.zero_grad(set_to_none=True)
            return {
                "loss": loss.item(),
                "loss_dict": {},
                "skipped": True,
                "needs_rollback": True,
                "grad_norm": grad_norm,
            }

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "loss_dict": {"train/MSE": loss.item()},
            "skipped": False,
            "needs_rollback": False,
            "grad_norm": grad_norm,
        }

    @staticmethod
    def _skip_result() -> Dict[str, Any]:
        return {
            "loss": float("nan"),
            "loss_dict": {},
            "skipped": True,
            "needs_rollback": False,
            "grad_norm": 0.0,
        }

    @torch.no_grad()
    def predict(self, snapshot: GNNSnapshot, n_steps: int = 1) -> torch.Tensor:
        """Autoregressive inference; returns the state after ``n_steps``."""
        self.model.eval()
        x = snapshot.x
        for _ in range(n_steps):
            x = self._forward(x)
        return x


# ----------------------------------------------------------------------------
# Fixed-horizon, single-target scheduling
# ----------------------------------------------------------------------------


class RolloutBuffer:
    """Accumulates a source snapshot and waits ``n_steps`` pushes for a
    single target ``n_steps`` steps later.

    ``n_steps`` is a horizon *distance*, not a window length: the loss is
    evaluated exactly once per training example, against the one real
    snapshot observed ``n_steps`` steps after the source — snapshots pushed
    in between are not kept or compared against. The model computation
    itself still rolls forward autoregressively through that ``n_steps``
    gap, feeding each step's own prediction back in (see
    `gnn.model.OnlineGNNTrainer.train_rollout_step`); this class only
    controls *when* the loss gets evaluated, not how many forward passes
    the model takes to get there. For a horizon that varies randomly each
    round instead of a fixed one, see :func:`sample_horizon` below.

    Pure bookkeeping over whatever snapshot object is pushed in — no
    `comin`/MPI dependency — so it is unit-testable standalone.
    """

    def __init__(self, n_steps: int) -> None:
        self.n_steps = int(n_steps)
        self.source: Optional[Any] = None
        self._pushes_since_source = 0

    def push(self, snapshot: Any) -> Optional[Tuple[Any, Any]]:
        """Push the latest snapshot.

        Returns ``(source, target)`` once ``n_steps`` pushes have occurred
        since the current source was set, else ``None``. After a pair is
        returned, the just-pushed snapshot becomes the *next* example's
        source — the horizon slides forward rather than resetting to empty.
        """
        if self.source is None:
            self.source = snapshot
            self._pushes_since_source = 0
            return None
        self._pushes_since_source += 1
        if self._pushes_since_source < self.n_steps:
            return None
        ready = (self.source, snapshot)
        self.source = snapshot
        self._pushes_since_source = 0
        return ready

    def reset(self) -> None:
        self.source = None
        self._pushes_since_source = 0

