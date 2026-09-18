"""Online Field-Space *Autoencoder* forecaster: compress each ICON timestep
into a multi-level latent, cache the latents in a reservoir, and predict the
next timestep from a *sequence* of cached latents.

Model levels and staged compression (`.claude/compression.png`)
---------------------------------------------------------------
``levels`` are R2B levels: the first is the *backbone*, kept as a field, and
the others are *residual* levels, e.g. ``levels = [4, 6, 7, 8]`` on an R2B8
run. Each timestep is split into the backbone mean and zero-mean residuals by
FieldSpaceNN's `encode_zooms` (paper eq 5). The encoder then folds the
finest residual level into the next coarser one, stage by stage, until
``latent_level`` is the finest level left; each stage is one
`FieldSpaceLayerConfig(type="mlp")` mapping a parent cell's own value plus
its 4 children to ``latent_channels`` values (paper eq 6), with ``n_blocks``
field-space attention blocks before and after:

    {x4, r6, r7, r8} --FSA--> --r8 into r7'--> {x4, r6, r7'} --FSA-->
    --r7' into r6'--> {x4, r6'} --FSA-->  latent                (latent_level=6)

The latent holds ``n_cells(latent_level) * latent_channels + n_cells(backbone)``
values, 0.254 of the native R2B8 field for the example with 4 channels. The
decoder mirrors the stages and `decode_zooms` sums the levels back to the fine
grid. Every layer is FieldSpaceNN's own (`MG_AutoEncoder`, attention blocks);
`FieldSpaceNN/` is not modified by this project.

FieldSpaceNN constraints this follows: grid layers exist for every level from
backbone to fine (see :mod:`icon_nested_mgrid`), since they are keyed by list
position; attention and compression blocks are ``block_type="ext"``, because
the legacy blocks reject zooms with different channel counts; attention
tokens live on the backbone level (``token_zoom`` must be the coarsest zoom),
so one token carries a backbone cell with all its descendants and attention
runs over backbone cells only (about 640 per rank for R2B4 on 32 GPUs).

Time
----
Every snapshot carries its ICON model time (unix seconds), and the reservoir
stores it with each latent. Every attention block is conditioned on it through
FieldSpaceNN's `TimeEmbedder` (sin/cos at periods of 1, 7, 30.4375 and 365.25
days, as in FieldSpaceNN's `configs/embedding/default.yaml`). The encoder uses
the snapshot's time, the processor window each cached latent's plus the
future slot's, and the decoder the time it decodes for. The spacing of the
future slot is measured from consecutive snapshot times.

The reservoir and the processor
-------------------------------
:class:`LatentReservoir` keeps the last ``n_history + rollout_steps - 1``
latents and their times, detached. The processor sees the ``n_history`` most
recent ones stacked on FieldSpaceNN's time axis ``t``, plus one future slot
filled with a copy of the latest latent (FieldSpaceNN's
``mask_ts_mode="repeat"`` convention) and stamped with the future valid time.
Its attention blocks use ``token_len_time = n_history + 1``, so the whole
window is one token per backbone cell. Its blocks are built with FieldSpaceNN's
block factory on the autoencoder's grid layers, because `MG_Transformer`
accepts only one channel count for all zooms. The blocks start near-identity
(FieldSpaceNN's ~1e-12 gates), so the untrained prediction is persistence.

What happens at ICON step t (``n_history = H``, ``rollout_steps = 1``)
--------------------------------------------------------------------
1. The reservoir holds z_{t-H} .. z_{t-1}.
2. x_t arrives. The prediction of x_t from z_{t-H} .. z_{t-1} is recomputed
   with gradient; the weights have not changed since it was written to ICON at
   step t-1, so it is the same number.
3. Loss = MSE(prediction of x_t, x_t) + ``recon_weight`` * MSE(decode(encode(
   x_t)), x_t), then one optimizer step.
4. z_t = encode(x_t), from step 3's forward pass, is cached; z_{t-H} drops out.
5. Predict x_{t+1} from z_{t-H+1} .. z_t and write it to ICON (plugin side).

With ``rollout_steps = n > 1``, step 2 rolls the processor n times from the
window ending at z_{t-n}.

The reconstruction term is required: cached latents are detached, so the
prediction loss cannot reach the encoder. A cached latent was produced by the
encoder as it was up to ``n_history + rollout_steps - 1`` optimizer steps ago;
that staleness is the price of never keeping past fine fields in memory.

Patches
-------
The trainer works on one rank's patch (:class:`icon_nested_mgrid.NestedMGrid`):
complete descendant trees of the rank's backbone cells, in nested order.
Patches of all ranks partition the globe, so the loss uses every cell.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from fieldspacenn.src.models.mg_autoencoder.mg_autoencoder import MG_AutoEncoder
from fieldspacenn.src.models.mg_transformer.mg_base_model import create_encoder_decoder_block
from fieldspacenn.src.modules.field_space.field_space_attention import (
    FieldSpaceAttentionConfig,
)
from fieldspacenn.src.modules.field_space.field_space_layer import FieldSpaceLayerConfig
from fieldspacenn.src.modules.grids.grid_utils import encode_zooms

from icon_nested_mgrid import MIN_BACKBONE_CELLS, NestedMGrid

# A latent (or a window of latents) in FieldSpaceNN's layout:
# {zoom: (b=1, v=1, t, n_zoom, d=nlev, f)}.
Latent = Dict[int, torch.Tensor]

_SECONDS_PER_DAY = 86400.0
# TimeEmbedder periods, in days (FieldSpaceNN configs/embedding/default.yaml).
_TIME_SCALES_DAYS = [1.0, 7.0, 30.4375, 365.25]


@dataclass
class FieldSpaceSnapshot:
    x_fine: torch.Tensor  # (n_fine, nlev) normalized, nested patch order, float32
    unix_seconds: float


# ----------------------------------------------------------------------------
# Reservoir
# ----------------------------------------------------------------------------


@dataclass
class LatentWindow:
    latent: Latent  # {zoom: (1, 1, length, n, d, f)}, oldest first
    unix_seconds: List[float]  # valid time of each state, oldest first


class LatentReservoir:
    """Ring buffer of latents from previous timesteps, one
    ``{zoom: (1, 1, 1, n, d, f)}`` dict plus its valid time per timestep,
    oldest first.

    Everything is stored detached and copied, so no autograd graph or
    in-place mutation reaches across timesteps. Not checkpointed. After a
    restart it refills within ``capacity`` steps.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self._buf: Deque[Tuple[Latent, float]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def push(self, latent: Latent, unix_seconds: float) -> None:
        """Append the newest state; the oldest drops out once full."""
        self._buf.append(
            ({zoom: t.detach().clone() for zoom, t in latent.items()}, float(unix_seconds))
        )

    @property
    def latest_seconds(self) -> Optional[float]:
        return self._buf[-1][1] if self._buf else None

    def window(self, length: int, lag: int = 0) -> Optional[LatentWindow]:
        """``length`` consecutive states stacked on the time axis (dim 2),
        oldest first, ending ``lag`` states before the most recent one.

        Returns ``None`` until enough states have accumulated. The caller
        skips the step rather than use a short or zero-padded history.
        """
        if length < 1 or lag < 0:
            raise ValueError(f"need length >= 1 and lag >= 0, got {length}, {lag}")
        if len(self._buf) < length + lag:
            return None
        end = len(self._buf) - lag
        entries = [self._buf[i] for i in range(end - length, end)]
        latent = {
            zoom: torch.cat([state[zoom] for state, _ in entries], dim=2) for zoom in entries[0][0]
        }
        return LatentWindow(latent=latent, unix_seconds=[seconds for _, seconds in entries])

    def numel(self) -> int:
        """Total cached elements: the memory the cache actually costs."""
        return sum(t.numel() for state, _ in self._buf for t in state.values())


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


@dataclass
class _Stage:
    """One compression stage: level ``fine`` folded into level ``parent``."""

    fine: int  # zoom
    parent: int  # zoom
    fine_features: int  # channels of ``fine`` before the stage


def compression_stages(levels: Sequence[int], latent_level: int) -> List[Tuple[int, int]]:
    """``(fine_level, parent_level)`` per stage, finest first: each residual
    level finer than ``latent_level`` is folded into the next coarser residual
    level."""
    residual = sorted(levels)[1:]
    return [
        (residual[i], residual[i - 1])
        for i in range(len(residual) - 1, 0, -1)
        if residual[i] > latent_level
    ]


class _LatentProcessor(nn.Module):
    """FieldSpaceNN attention blocks over the latent zooms, built on shared
    grid layers with per-zoom channel counts."""

    def __init__(self, blocks: Dict[str, nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleDict(blocks)

    def forward(self, x_zooms_groups, emb_groups, sample_configs):
        for block in self.blocks.values():
            x_zooms_groups = block(x_zooms_groups, sample_configs=sample_configs, emb_groups=emb_groups)
        return x_zooms_groups


class FieldSpaceAEForecaster(nn.Module):
    """Encoder/decoder (`MG_AutoEncoder`) plus temporal processor over one
    rank's nested patch, all time-conditioned. See the module docstring.

    :meth:`forward` is the single training entry point (one call per
    backward, so DDP's reducer sees every parameter used exactly once per
    iteration); :meth:`encode`, :meth:`decode` and :meth:`rollout` are the
    building blocks, also used directly for inference. Times are unix seconds.
    """

    def __init__(
        self,
        mgrid: NestedMGrid,
        nlev: int,
        levels: Sequence[int],
        latent_level: int,
        n_history: int,
        latent_channels: int = 4,
        n_blocks: int = 1,
        n_processor_blocks: int = 2,
        att_dim: int = 32,
        n_head_channels: int = 8,
        hidden_dim: int = 64,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        levels = sorted(int(level) for level in levels)
        if len(levels) < 2 or len(set(levels)) != len(levels):
            raise ValueError(f"need a backbone and at least one residual level, got {levels}")
        if levels[0] != mgrid.base_level or levels[-1] != mgrid.fine_level:
            raise ValueError(
                f"levels {levels} must start at the patch's base level R2B{mgrid.base_level} "
                f"and end at its fine level R2B{mgrid.fine_level}"
            )
        if latent_level not in levels[1:]:
            raise ValueError(f"latent_level {latent_level} must be one of the residual levels {levels[1:]}")

        self.mgrid = mgrid
        self.nlev = int(nlev)
        self.levels = levels
        self.latent_level = int(latent_level)
        self.n_history = int(n_history)
        self.latent_channels = int(latent_channels)
        self.fine_zoom = mgrid.zoom(mgrid.fine_level)
        self.all_zooms = list(range(self.fine_zoom + 1))
        in_zooms = [mgrid.zoom(level) for level in levels]
        backbone_zoom = in_zooms[0]

        time_embed_confs = {
            "embed_names": ["TimeEmbedder"],
            "embed_mode": "sum",
            "embed_confs": {
                "TimeEmbedder": {
                    "in_channels": 1,
                    "embed_dim": time_embed_dim,
                    "time_scales": _TIME_SCALES_DAYS,
                    # time_min/time_max only scale the linear-trend branch, which is off.
                    "time_min": 0.0,
                    "time_max": 1.0,
                    "use_linear": False,
                    # One embedder per block. FieldSpaceNN's shared-embedder cache is
                    # a process-wide singleton keyed on object ids, so sharing could
                    # hand a later model a stale embedder.
                    "shared": False,
                }
            },
        }

        def attention(zooms: List[int], token_len_time: int = 1) -> FieldSpaceAttentionConfig:
            return FieldSpaceAttentionConfig(
                token_zoom=backbone_zoom,
                q_zooms=list(zooms),
                kv_zooms=list(zooms),
                att_dim=att_dim,
                n_head_channels=n_head_channels,
                token_len_time=token_len_time,
                embed_confs=time_embed_confs,
                block_type="ext",
            )

        def field_layer(in_z, target_z, field_z, out_z, target_features) -> FieldSpaceLayerConfig:
            return FieldSpaceLayerConfig(
                in_zooms=in_z,
                target_zooms=target_z,
                field_zoom=field_z,
                out_zooms=out_z,
                target_features=target_features,
                type="mlp",
                block_type="ext",
                hidden_dim=hidden_dim,
            )

        features = {zoom: 1 for zoom in in_zooms}
        zooms = list(in_zooms)
        self.stages: List[_Stage] = []
        encoder_blocks: Dict[str, Any] = {}
        for i in range(n_blocks):
            encoder_blocks[f"att_in_{i}"] = attention(zooms)
        for s, (fine_level, parent_level) in enumerate(compression_stages(levels, latent_level)):
            stage = _Stage(mgrid.zoom(fine_level), mgrid.zoom(parent_level), features[mgrid.zoom(fine_level)])
            self.stages.append(stage)
            zooms = [zoom for zoom in zooms if zoom != stage.fine]
            encoder_blocks[f"compress_{s}"] = field_layer(
                [stage.parent, stage.fine], [stage.parent], stage.parent, zooms, self.latent_channels
            )
            features[stage.parent] = self.latent_channels
            del features[stage.fine]
            for i in range(n_blocks):
                encoder_blocks[f"att_{s}_{i}"] = attention(zooms)

        self.latent_zooms: List[int] = list(zooms)
        self.latent_features: Dict[int, int] = dict(features)

        decoder_blocks: Dict[str, Any] = {}
        for i in range(n_blocks):
            decoder_blocks[f"latent_att_{i}"] = attention(zooms)
        for s in reversed(range(len(self.stages))):
            stage = self.stages[s]
            zooms = sorted(zooms + [stage.fine])
            decoder_blocks[f"decompress_{s}"] = field_layer(
                [stage.parent], [stage.parent, stage.fine], stage.parent, zooms,
                {stage.parent: 1, stage.fine: stage.fine_features},
            )
            for i in range(n_blocks):
                decoder_blocks[f"att_{s}_{i}"] = attention(zooms)

        self.autoencoder = MG_AutoEncoder(
            mgrids=mgrid.fieldspace_mgrids(),
            in_zooms=in_zooms,
            encoder_block_configs=encoder_blocks,
            decoder_block_configs=decoder_blocks,
            in_features=1,
            n_head_channels=n_head_channels,
        )
        assert list(self.autoencoder.bottleneck_zooms) == self.latent_zooms

        latent_feature_list = [self.latent_features[zoom] for zoom in self.latent_zooms]
        self.processor = _LatentProcessor(
            {
                f"att_{i}": create_encoder_decoder_block(
                    attention(self.latent_zooms, token_len_time=self.n_history + 1),
                    self.latent_zooms,
                    latent_feature_list,
                    [1],
                    grid_layers=self.autoencoder.grid_layers,
                    n_head_channels=n_head_channels,
                )
                for i in range(n_processor_blocks)
            }
        )

        # Whole patch as one sample. Past/future counts only have to be
        # consistent between zooms within a call: one timestep for
        # encode/decode, n_history past + 1 future slot for the processor.
        self._sample_configs_step = {
            zoom: {"n_past_ts": 1, "n_future_ts": 0, "zoom_patch_sample": -1, "mask_n_last_ts": 0}
            for zoom in self.all_zooms
        }
        self._sample_configs_window = {
            zoom: {
                "n_past_ts": self.n_history, "n_future_ts": 1,
                "zoom_patch_sample": -1, "mask_n_last_ts": 1,
            }
            for zoom in self.all_zooms
        }

    def latent_numel(self) -> int:
        """Latent values per timestep."""
        return sum(
            self.mgrid.n_cells(self.mgrid.base_level + zoom) * self.nlev * self.latent_features[zoom]
            for zoom in self.latent_zooms
        )

    def _time_emb(self, unix_seconds: Sequence[float]) -> Dict[str, Any]:
        """FieldSpaceNN embedding input for the given valid times (one per
        timestep on the ``t`` axis): ``{"TimeEmbedder": {zoom: (1, t) days}}``.

        Days since the unix epoch as float32, like FieldSpaceNN's dataset
        times (about 2-minute resolution). A fresh dict every call, because
        FieldSpaceNN blocks add keys to it in place.
        """
        days = torch.tensor(
            [[float(s) / _SECONDS_PER_DAY for s in unix_seconds]],
            dtype=torch.float32,
            device=self.autoencoder.zooms.device,
        )
        return {"TimeEmbedder": {zoom: days for zoom in self.all_zooms}}

    def to_pyramid(self, x_fine: torch.Tensor) -> Latent:
        """(n_fine, nlev) -> {backbone: mean, residual levels: zero-mean
        residual against the next coarser model level}, in (b, v, t, n, d, f)
        layout, via FieldSpaceNN's `encode_zooms`."""
        n_backbone, fine_zoom = self.mgrid.n_backbone, self.fine_zoom
        x = x_fine.reshape(n_backbone, 4**fine_zoom, self.nlev)
        means = {}
        for level in self.levels:
            zoom = self.mgrid.zoom(level)
            n = n_backbone * 4**zoom
            # mean() allocates: encode_zooms subtracts in place and must not
            # alias the caller's snapshot.
            means[zoom] = x.reshape(n, 4 ** (fine_zoom - zoom), self.nlev).mean(dim=1).reshape(
                1, 1, 1, n, self.nlev, 1
            )
        return encode_zooms(means, self._sample_configs_step, None)

    def encode(self, x_fine: torch.Tensor, unix_seconds: float) -> Latent:
        return self.autoencoder.ae_encode(
            [self.to_pyramid(x_fine)],
            sample_configs=self._sample_configs_step,
            emb_groups=[self._time_emb([unix_seconds])],
        )[0]

    def decode(self, latent: Latent, unix_seconds: float) -> torch.Tensor:
        out = self.autoencoder.ae_decode(
            [dict(latent)],
            sample_configs=self._sample_configs_step,
            emb_groups=[self._time_emb([unix_seconds])],
            out_zoom=self.fine_zoom,
        )[0][self.fine_zoom]
        return out.reshape(self.mgrid.n_fine, self.nlev)

    def step(self, window: Latent, window_seconds: Sequence[float], next_seconds: float) -> Latent:
        """One processor step: ``n_history`` states at ``window_seconds`` in,
        the state at ``next_seconds`` out."""
        x = {zoom: torch.cat([w, w[:, :, -1:]], dim=2) for zoom, w in window.items()}
        out = self.processor(
            x_zooms_groups=[x],
            emb_groups=[self._time_emb([*window_seconds, next_seconds])],
            sample_configs=self._sample_configs_window,
        )[0]
        return {zoom: out[zoom][:, :, -1:] for zoom in self.latent_zooms}

    def rollout(
        self, window: Latent, window_seconds: Sequence[float], n_steps: int, step_seconds: float
    ) -> Latent:
        """``n_steps`` autoregressive processor steps, ``step_seconds`` apart;
        each prediction slides into the window and the oldest state drops out.
        Returns the state at ``window_seconds[-1] + n_steps * step_seconds``."""
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        seconds = list(window_seconds)
        for _ in range(n_steps):
            next_seconds = seconds[-1] + step_seconds
            nxt = self.step(window, seconds, next_seconds)
            window = {zoom: torch.cat([window[zoom][:, :, 1:], nxt[zoom]], dim=2) for zoom in nxt}
            seconds = seconds[1:] + [next_seconds]
        return nxt

    def forward(
        self,
        x_now: torch.Tensor,
        now_seconds: float,
        window: Latent,
        window_seconds: Sequence[float],
        n_steps: int,
        step_seconds: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, Latent]:
        """Training pass. Returns ``(reconstruction of x_now, prediction of
        x_now from the window, latent of x_now)``; the window must end
        ``n_steps * step_seconds`` before ``now_seconds``."""
        latent_now = self.encode(x_now, now_seconds)
        recon = self.decode(latent_now, now_seconds)
        pred = self.decode(self.rollout(window, window_seconds, n_steps, step_seconds), now_seconds)
        return recon, pred, latent_now


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------


class OnlineFieldSpaceAETrainer:
    """DDP-wrapped :class:`FieldSpaceAEForecaster` trainer with a
    :class:`LatentReservoir`, performing online training on one rank's patch.

    Every call to :meth:`train_step` caches the snapshot's latent, and trains
    once the reservoir holds ``n_history + rollout_steps - 1`` earlier
    latents. Every rank pushes the same number of snapshots, so all ranks
    leave warm-up on the same step and call backward together, as DDP
    requires. A skip on only some ranks (a NaN input) breaks that lockstep.
    That risk is known and not handled here.
    """

    def __init__(
        self,
        nlev: int,
        mgrid: NestedMGrid,
        levels: Sequence[int],
        latent_level: int,
        n_history: int = 4,
        rollout_steps: int = 1,
        latent_channels: int = 4,
        n_blocks: int = 1,
        n_processor_blocks: int = 2,
        recon_weight: float = 1.0,
        lr: float = 2e-4,
        att_dim: int = 32,
        n_head_channels: int = 8,
        hidden_dim: int = 64,
        time_embed_dim: int = 64,
        grad_clip: Optional[float] = 1.0,
        use_ddp: Optional[bool] = None,
        device: Optional[torch.device] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        rank: Optional[int] = None,
    ) -> None:
        if n_history < 1 or rollout_steps < 1:
            raise ValueError(
                f"need n_history >= 1 and rollout_steps >= 1, got {n_history}, {rollout_steps}"
            )
        if mgrid.n_backbone < MIN_BACKBONE_CELLS:
            raise ValueError(
                f"patch has {mgrid.n_backbone} backbone cells, need at least {MIN_BACKBONE_CELLS}"
            )
        self.nlev = int(nlev)
        self.mgrid = mgrid
        self.n_history = int(n_history)
        self.rollout_steps = int(rollout_steps)
        self.recon_weight = float(recon_weight)
        self.device = torch.device("cuda", 0) if device is None else device
        self.grad_clip = grad_clip
        self.rank = rank
        self.log_fn = log_fn if log_fn is not None else (lambda msg: None)

        # Training needs the window that ends rollout_steps - 1 states before
        # the newest cached one (the target is the snapshot being pushed now);
        # prediction needs only the newest n_history.
        self.reservoir = LatentReservoir(capacity=self.n_history + self.rollout_steps - 1)
        # Time between consecutive snapshots, measured from their ICON times.
        # Unknown until the second snapshot.
        self.step_seconds: Optional[float] = None

        # FieldSpaceNN creates device-less tensors inside GridLayer.__init__
        # and during forward passes; this puts them on the right device.
        with torch.device(self.device):
            self.model = FieldSpaceAEForecaster(
                mgrid,
                nlev=self.nlev,
                levels=levels,
                latent_level=latent_level,
                n_history=self.n_history,
                latent_channels=latent_channels,
                n_blocks=n_blocks,
                n_processor_blocks=n_processor_blocks,
                att_dim=att_dim,
                n_head_channels=n_head_channels,
                hidden_dim=hidden_dim,
                time_embed_dim=time_embed_dim,
            ).to(self.device)
        self.forward_model = self._wrap_ddp(use_ddp)
        self.optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=lr, weight_decay=0.0)

        n_params = sum(p.numel() for p in self.model.parameters())
        self.log_fn(
            f"[rank={rank}] FieldSpace AE trainer initialized: levels={self.model.levels}, "
            f"latent_level={latent_level}, latent_zooms={self.model.latent_zooms} "
            f"(latent/native size={self.model.latent_numel() / (mgrid.n_fine * self.nlev):.3f}), "
            f"nlev={nlev}, n_history={n_history}, rollout_steps={rollout_steps}, "
            f"reservoir_capacity={self.reservoir.capacity}, n_backbone={mgrid.n_backbone}, "
            f"n_fine={mgrid.n_fine}, params={n_params:,}, device={self.device}, "
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
        # broadcast_buffers=False: grid-layer buffers differ in shape between
        # ranks (patch sizes differ) and must not be synchronized.
        return DDP(
            self.model,
            device_ids=[self.device.index or 0],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    @torch.no_grad()
    def prepare_snapshot(self, x_fine: torch.Tensor, unix_seconds: float) -> FieldSpaceSnapshot:
        x_fine = x_fine.to(self.device, dtype=torch.float32, non_blocking=True).detach()
        return FieldSpaceSnapshot(x_fine=x_fine, unix_seconds=float(unix_seconds))

    @staticmethod
    def _skip_result(needs_rollback: bool = False, loss: float = float("nan")) -> Dict[str, Any]:
        return {
            "loss": loss,
            "loss_dict": {},
            "skipped": True,
            "needs_rollback": needs_rollback,
            "grad_norm": 0.0,
        }

    def _cache(self, latent: Latent, unix_seconds: float) -> None:
        # A non-finite latent means the model itself is broken. Restart the
        # history rather than leave a gap: a gap would silently shift which
        # timesteps the window's positions correspond to.
        if all(torch.isfinite(t).all() for t in latent.values()):
            self.reservoir.push(latent, unix_seconds)
        else:
            self.reservoir.clear()

    def train_step(self, snapshot: FieldSpaceSnapshot) -> Dict[str, Any]:
        """Train on ``snapshot`` as target (once the reservoir is warm), then
        cache its latent. See "What happens at ICON step t" in the module
        docstring.

        Loss = MSE(prediction, x) + ``recon_weight`` * MSE(reconstruction, x)
        over all patch cells. The prediction is decoded from ``rollout_steps``
        processor steps starting at the ``n_history`` cached latents that end
        ``rollout_steps`` steps before ``snapshot``.

        A NaN/Inf snapshot is neither trained on nor cached. The reservoir is
        cleared instead, to keep the window contiguous in time.
        """
        if not torch.isfinite(snapshot.x_fine).all():
            self.log_fn("[trainer] NaN/Inf in snapshot — skipping step, clearing reservoir")
            self.reservoir.clear()
            return self._skip_result()

        x, now = snapshot.x_fine, snapshot.unix_seconds
        if self.reservoir.latest_seconds is not None:
            self.step_seconds = now - self.reservoir.latest_seconds

        window = self.reservoir.window(self.n_history, lag=self.rollout_steps - 1)
        if window is None:
            with torch.no_grad(), torch.device(self.device):
                self.model.eval()
                self._cache(self.model.encode(x, now), now)
            return self._skip_result()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.device(self.device):
            recon, pred, latent_now = self.forward_model(
                x, now, window.latent, window.unix_seconds, self.rollout_steps, self.step_seconds
            )
        loss_pred = torch.mean((pred - x) ** 2)
        loss_recon = torch.mean((recon - x) ** 2)
        loss = loss_pred + self.recon_weight * loss_recon

        if not torch.isfinite(loss):
            self.log_fn(
                f"[trainer] Non-finite loss ({loss.item()}) — model may be corrupted"
            )
            self.reservoir.clear()
            return self._skip_result(needs_rollback=True, loss=loss.item())

        loss.backward()

        max_norm = self.grad_clip if self.grad_clip is not None else float("inf")
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm).item()
        if not math.isfinite(grad_norm):
            self.log_fn(
                f"[trainer] Non-finite grad_norm ({grad_norm}) — skipping optimizer step"
            )
            self.optimizer.zero_grad(set_to_none=True)
            self.reservoir.clear()
            result = self._skip_result(needs_rollback=True, loss=loss.item())
            result["grad_norm"] = grad_norm
            return result

        self.optimizer.step()
        # Cached with the encoder as it was before this update. See the module
        # docstring on staleness.
        self._cache(latent_now, now)

        return {
            "loss": loss.item(),
            "loss_dict": {
                "train/MSE_pred": loss_pred.item(),
                "train/MSE_recon": loss_recon.item(),
            },
            "skipped": False,
            "needs_rollback": False,
            "grad_norm": grad_norm,
        }

    @torch.no_grad()
    def predict(self, n_steps: int = 1) -> Optional[torch.Tensor]:
        """Forecast the fine field ``n_steps`` steps after the most recently
        cached latent, from the newest ``n_history`` cached latents.

        Returns ``(n_fine, nlev)`` normalized in nested patch order, or
        ``None`` while the reservoir holds fewer than ``n_history`` latents
        or before the step length is known (second snapshot).
        """
        if self.step_seconds is None:
            return None
        window = self.reservoir.window(self.n_history)
        if window is None:
            return None
        self.model.eval()
        valid_seconds = window.unix_seconds[-1] + n_steps * self.step_seconds
        with torch.device(self.device):
            latent = self.model.rollout(window.latent, window.unix_seconds, n_steps, self.step_seconds)
            return self.model.decode(latent, valid_seconds)
