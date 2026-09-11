"""Online FieldSpaceNN (Field-Space Transformer) trainer, operating on the
compact two-zoom multi-grid built by :mod:`icon_mgrid_utils`.

Mirrors :mod:`gnn_online`'s plugin-facing shape (`*Snapshot`
dataclass, `Online*Trainer.prepare_snapshot`/`train_rollout_step`/
`predict`, `_wrap_ddp`) so `fieldspace_plugin.py` can follow
`gnn_plugin.py`'s control flow almost verbatim. The model itself is
FieldSpaceNN's `MG_Transformer` (imported as an installed library, not
copied -- `FieldSpaceNN/` is not modified by this project) instead of a
hand-rolled GNN.

Deliberately minimal for v1 (per explicit project decision -- see the plan
document written for this task): a single `FieldSpaceAttentionConfig`
block attending jointly over both zooms, no `ConservativeLayerConfig`, a
single scalar feature per cell. Config values (`att_dim`, `n_head_channels`,
and the `sample_configs` entries) match the one proven-working reference
for this model, the user-provided offline demo `icon_test.py`, rather than
untested guesses -- deviating only where v1 is deliberately simpler (one
attention block instead of two, no conservative layer).

The model predicts the next field state directly (FieldSpaceNN's residual
updates are internal to each block, not an external "input + tendency"
step the caller has to apply -- confirmed by `icon_test.py` comparing the
model's raw output directly against the target field), unlike
`gnn_online.GNNModel`, which does add its own external residual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from fieldspacenn.src.modules.field_space.field_space_attention import (
    FieldSpaceAttentionConfig,
)
from fieldspacenn.src.models.mg_transformer.mg_transformer import MG_Transformer

from icon_mgrid_utils import LocalMGrid, pool_fine_to_coarse

# Online normalization (RunningMeanStd) and fixed-horizon scheduling
# (RolloutBuffer) are generic, model-agnostic bookkeeping, shared with
# gnn_plugin.py/unet_plugin.py -- fieldspace_plugin.py imports them from
# MEssE.utils.icon_online_helper directly, not from this module.

_COARSE_ZOOM = 0
_FINE_ZOOM = 1

# FieldSpaceNN's GridLayer.__init__ fills any masked adjacency slot (column
# k in {1,3,4,5,7,8}) with the raw integer value `k - 1` (up to 8) as a
# "consistent fallback direction" -- not a self-fallback, and not
# necessarily in-bounds. HEALPix grids never hit this (always far more than
# 9 pixels at any zoom actually used), but our per-rank coarse zoom can be
# small near partition edges, and indexing `coordinates[adjc]` with a
# fallback value >= n_cells raises IndexError. A rank whose local grid would
# be smaller than this must skip the step entirely (see
# OnlineFieldSpaceTrainer's n_coarse guard) rather than let GridLayer
# construct on too few cells. Discovered via this trainer's own unit tests
# on a small synthetic grid, not merely a test-scale artifact -- see the
# plan document's note on this finding.
_MIN_ZOOM_CELLS = 9


@dataclass
class FieldSpaceSnapshot:
    x_fine: torch.Tensor  # (4*n_coarse, nlev) normalized, compact order, float32, GPU
    unix_seconds: float


class OnlineFieldSpaceTrainer:
    """DDP-wrapped MG_Transformer trainer performing rollout-based online
    training on one rank's compact two-zoom local patch.

    Each rank's `n_coarse`/`4*n_coarse` differ (domain decomposition varies
    the number of complete coarse quad-groups per rank), but the model's
    parameters are shape-identical across ranks, so standard DDP gradient
    all-reduce applies unmodified -- same argument as
    `gnn_online.OnlineGNNTrainer`. One new case this design introduces: a
    rank can have too few complete quad-groups (`n_coarse < _MIN_ZOOM_CELLS`,
    including the `n_coarse == 0` extreme -- e.g. a small node count or an
    unlucky partition) -- such a rank must skip model construction and every
    training/prediction step entirely, not just call the model on an empty
    batch (see `_usable`, `train_rollout_step`/`predict`).
    """

    def __init__(
        self,
        nlev: int,
        mgrid: LocalMGrid,
        lr: float = 2e-4,
        att_dim: int = 32,
        n_head_channels: int = 8,
        grad_clip: Optional[float] = 1.0,
        use_ddp: Optional[bool] = None,
        device: Optional[torch.device] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        rank: Optional[int] = None,
    ) -> None:
        self.nlev = int(nlev)
        self.mgrid = mgrid
        self.att_dim = att_dim
        self.n_head_channels = n_head_channels
        self.device = torch.device("cuda", 0) if device is None else device
        self.grad_clip = grad_clip
        self.rank = rank
        self.log_fn = log_fn if log_fn is not None else (lambda msg: None)

        # Whole local patch as one "patch" per zoom, single past/future
        # timestep -- matches icon_test.py's own (proven-working)
        # sample_configs exactly, since deeper masking semantics
        # (mask_n_last_ts) are not otherwise exercised/verified here.
        self._sample_configs = {
            _COARSE_ZOOM: {
                "n_past_ts": 1, "n_future_ts": 1, "zoom_patch_sample": -1, "mask_n_last_ts": 1,
            },
            _FINE_ZOOM: {
                "n_past_ts": 1, "n_future_ts": 1, "zoom_patch_sample": -1, "mask_n_last_ts": 1,
            },
        }

        # GridLayer.__init__ fills masked adjacency slots with raw
        # column-index values up to 8 (see _MIN_ZOOM_CELLS docstring above)
        # -- out of bounds, and a hard crash, for a zoom with fewer than
        # that many cells. A too-small coarse zoom (and therefore too-small
        # fine zoom, always >= 4x as large) must skip model construction
        # entirely, not just the training/prediction calls.
        self._usable = mgrid.n_coarse >= _MIN_ZOOM_CELLS
        if not self._usable:
            self.model = None
            self.forward_model = None
            self.optimizer = None
            self.log_fn(
                f"[rank={rank}] FieldSpace trainer NOT built: n_coarse={mgrid.n_coarse} "
                f"< {_MIN_ZOOM_CELLS} (GridLayer's masked-neighbor fallback needs at "
                f"least that many cells per zoom) — this rank will skip training/"
                f"prediction every step until a rebuild with more complete quad-groups."
            )
            return

        self.model = self._build_model().to(self.device)
        self.forward_model = self._wrap_ddp(use_ddp)
        self.optimizer = torch.optim.Adam(
            self.forward_model.parameters(), lr=lr, weight_decay=0.0
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        self.log_fn(
            f"[rank={rank}] FieldSpace trainer initialized: nlev={nlev}, "
            f"att_dim={att_dim}, n_head_channels={n_head_channels}, "
            f"n_coarse={mgrid.n_coarse}, n_fine_kept={mgrid.n_fine_kept}, "
            f"params={n_params:,}, device={self.device}, "
            f"ddp={isinstance(self.forward_model, DDP)}"
        )

    def _build_model(self) -> MG_Transformer:
        mgrids = [
            {
                "coords": self.mgrid.coarse_coords,
                "adjc": self.mgrid.coarse_adjc,
                "adjc_mask": self.mgrid.coarse_adjc_mask,
                "zoom": _COARSE_ZOOM,
            },
            {
                "coords": self.mgrid.fine_coords,
                "adjc": self.mgrid.fine_adjc,
                "adjc_mask": self.mgrid.fine_adjc_mask,
                "zoom": _FINE_ZOOM,
            },
        ]
        block_configs = {
            "0": FieldSpaceAttentionConfig(
                token_zoom=_COARSE_ZOOM,
                q_zooms=[_COARSE_ZOOM, _FINE_ZOOM],
                kv_zooms=[_COARSE_ZOOM, _FINE_ZOOM],
                att_dim=self.att_dim,
                n_head_channels=self.n_head_channels,
            ),
            # No "conservative" ConservativeLayerConfig entry in v1 -- a
            # one-line, backward-compatible v2 addition once the base model
            # trains end-to-end (deliberately deferred, see module
            # docstring / the plan document).
        }
        # GridLayer.__init__ (FieldSpaceNN, not modified) creates several
        # derived-statistics tensors (dist_quantiles, nh_dist, min_dist,
        # ...) without an explicit device -- e.g. `dists.quantile(
        # torch.linspace(0.01, 0.99, 20))`, where `torch.linspace(...)`
        # defaults to CPU. Since we feed it GPU-resident coords (self.mgrid
        # is built on self.device), that raises "quantile() q tensor must
        # be on the same device as the input tensor" (found on a real GPU
        # run, job 27393947). Those statistics are plain tensor attributes,
        # not registered buffers, so a later `.to(self.device)` would not
        # move them even if construction succeeded on CPU first -- the
        # robust fix is to make every device-less tensor FieldSpaceNN
        # creates during construction default to `self.device` in the
        # first place, via torch's default-device context manager, rather
        # than patching this one call site (there may be others like it).
        with torch.device(self.device):
            return MG_Transformer(
                mgrids=mgrids,
                block_configs=block_configs,
                in_zooms=[_COARSE_ZOOM, _FINE_ZOOM],
                in_features=1,
                n_groups_variables=[1],
                n_head_channels=self.n_head_channels,
            )

    def _wrap_ddp(self, use_ddp: Optional[bool]) -> nn.Module:
        if use_ddp is None:
            use_ddp = dist.is_available() and dist.is_initialized()
        if not use_ddp:
            return self.model
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size <= 1:
            return self.model
        return DDP(
            self.model,
            device_ids=[self.device.index or 0],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    def _to_zoom_tensors(self, x_fine_compact: torch.Tensor):
        """(4*n_coarse, nlev) -> the {zoom: (b,v,t,n,d,f)} layout
        MG_Transformer expects (nlev folded into the depth axis `d`, a
        single variable/feature, whole local patch as one token/patch)."""
        n_coarse = self.mgrid.n_coarse
        coarse = pool_fine_to_coarse(x_fine_compact, n_coarse)  # (n_coarse, nlev)
        fine_t = x_fine_compact.view(1, 1, 1, -1, self.nlev, 1)
        coarse_t = coarse.view(1, 1, 1, -1, self.nlev, 1)
        return [{_COARSE_ZOOM: coarse_t, _FINE_ZOOM: fine_t}]

    def _forward(self, x_fine_compact: torch.Tensor) -> torch.Tensor:
        x_zooms_groups = self._to_zoom_tensors(x_fine_compact)
        # Same default-device guard as _build_model (see its comment): the
        # forward pass can equally well create device-less tensors inside
        # FieldSpaceNN (e.g. positional/embedding machinery), not just
        # __init__ -- cheap to guard here too rather than risk discovering
        # a second instance of the same bug class only via another GPU run.
        with torch.device(self.device):
            out = self.forward_model(
                x_zooms_groups=x_zooms_groups,
                emb_groups=[{}],
                sample_configs=self._sample_configs,
                out_zoom=_FINE_ZOOM,
            )
        return out[0][_FINE_ZOOM].reshape(self.mgrid.n_fine_kept, self.nlev)

    @torch.no_grad()
    def prepare_snapshot(self, x_fine: torch.Tensor, unix_seconds: float) -> FieldSpaceSnapshot:
        x_fine = x_fine.to(self.device, dtype=torch.float32, non_blocking=True).detach()
        return FieldSpaceSnapshot(x_fine=x_fine, unix_seconds=float(unix_seconds))

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self.mgrid.fine_compact_owned_mask.unsqueeze(-1)
        diff2 = (pred - target) ** 2 * mask
        return diff2.sum() / (mask.sum() * pred.shape[-1]).clamp(min=1)

    def train_rollout_step(
        self,
        source: FieldSpaceSnapshot,
        target: FieldSpaceSnapshot,
        n_steps: int,
    ) -> Dict[str, Any]:
        """Roll the model forward autoregressively for ``n_steps`` steps
        from ``source`` (each step re-pools the coarse zoom from the
        model's own latest fine-zoom prediction -- the coarse zoom is never
        carried forward as independent state), then evaluate the loss once
        against ``target``, the single ground-truth snapshot observed
        ``n_steps`` steps after ``source``. Loss only considers owned
        (non-halo) fine cells that are also part of a complete coarse
        quad-group, matching `icon_mgrid_utils`'s compact ordering.

        Skips the step entirely (no forward pass) if this rank's coarse
        zoom is too small for the model to have been built at all (see
        `_usable`/`_MIN_ZOOM_CELLS`) -- unlike the GNN trainer, which never
        has a genuinely empty/too-small rank.
        """
        if not self._usable:
            self.log_fn("[trainer] model not built (n_coarse too small) — skipping rollout step")
            return self._skip_result()
        if not torch.isfinite(source.x_fine).all():
            self.log_fn("[trainer] NaN/Inf in source — skipping rollout step")
            return self._skip_result()
        if not torch.isfinite(target.x_fine).all():
            self.log_fn("[trainer] NaN/Inf in target — skipping rollout step")
            return self._skip_result()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        x = source.x_fine
        for _ in range(n_steps):
            x = self._forward(x)
        loss = self._masked_mse(x, target.x_fine)

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
    def predict(self, snapshot: FieldSpaceSnapshot, n_steps: int = 1) -> torch.Tensor:
        """Autoregressive inference; returns the fine-zoom state after
        ``n_steps``. Returns the input unchanged if this rank's model was
        never built (`n_coarse` too small, see `_usable`)."""
        if not self._usable:
            return snapshot.x_fine
        self.model.eval()
        x = snapshot.x_fine
        for _ in range(n_steps):
            x = self._forward(x)
        return x
