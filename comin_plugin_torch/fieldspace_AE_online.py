"""Online Field-Space *Autoencoder* forecaster: compress each ICON timestep,
cache the compressed states in a reservoir, and predict the next timestep
from a *sequence* of cached compressed states.

Sibling of :mod:`fieldspace_online` (same two-zoom local multi-grid from
:mod:`icon_mgrid_utils`, same plugin-facing shape: `prepare_snapshot`,
`train_step`/`predict`, `_wrap_ddp`), so `fieldspace_AE_plugin.py` can
follow `fieldspace_plugin.py`'s control flow. The difference is the model:

    x_t (fine) --mg-tokenize--> {x_coarse, r_fine}          (encode_zooms)
               --FSA x k--> --compress--> --FSA x k--> z_t  (encoder)
    z_{t-H..t-1} (reservoir) --temporal FSA--> z_t          (processor)
    z_t        --FSA x k--> --decompress--> --FSA x k-->
               {x_coarse, r_fine} --decode_zooms--> x_t     (decoder)

Every layer is FieldSpaceNN's own, imported not copied (`FieldSpaceNN/` is
not modified by this project): the encoder/decoder is `MG_AutoEncoder`, the
model class of Meuer et al. 2026 ("Field-space autoencoder for scalable
climate emulators"), with `FieldSpaceLayerConfig(type="mlp")` as the
compression/decompression block exactly as in FieldSpaceNN's shipped
`configs/model/mg_autoencoder.yaml`; the processor is an `MG_Transformer`.

Mapping the design sketch (`.claude/compression.png`, drawn for R2B8 with
pyramid {3, 6, 7, 8}) onto the two-zoom multi-grid
------------------------------------------------------------------------
`icon_mgrid_utils` builds two zooms: the grid ICON runs on (fine, e.g. R2B4
or R2B8) and the grid one refinement level up (coarse, R2B3 or R2B7). With
only two zooms there is a single compression stage, and its target is the
coarse level itself: the patch per coarse cell is its own mean plus its 4
fine residual children (``1 + 4 = 5`` values, paper eq 5), mapped to
``latent_channels`` values (paper eq 6). The latent is one zoom-0 tensor, so
per timestep it holds ``latent_channels / 4`` as many values as the native
field:

    latent_channels = 4  -> 1x   (5 -> 4: exactly the pyramid's free DOF,
                                  so reconstruction can be lossless)
    latent_channels = 2  -> 1/2
    latent_channels = 1  -> 1/4

The sketch's R2B8 memory saving (0.26x) comes from folding R2B8 all the way
down to R2B3, which needs a deeper multi-grid than `icon_mgrid_utils` builds
today.

Every attention block attends over all coarse cells of the rank's patch at
once (``zoom_patch_sample=-1``). PyTorch's SDPA keeps the memory linear in
that count, but compute grows with its square: about 1.3k cells per rank for
R2B4 on 4 GPUs, about 82k for R2B8 on 16 GPUs.
Two FieldSpaceNN constraints matter when going there: `MG_base_model` keys
`GridLayer`s by *list position*, so every intermediate refinement level must
be present in the mgrid list (zoom labels must be consecutive); and the legacy
`FieldSpaceAttentionBlock` rejects zooms with different channel counts (e.g.
{x3: 1, r7': 4}), which needs ``block_type="ext"``.

Why the multi-scale input is residual, not raw
----------------------------------------------
The fine zoom is fed as a zero-mean residual against its coarse parent
(FieldSpaceNN's `encode_zooms`, the same call its dataset pipeline makes), so
`decode_zooms`' upsample-and-sum reconstructs the field exactly. Feeding the
raw fine field at zoom 1 instead would make the untrained model output
``x + mean(x)``.

Time
----
Every snapshot carries its ICON model time (``comin.current_get_datetime()``,
as unix seconds), and the reservoir stores it with each latent. Every
attention block (encoder, decoder, processor) is conditioned on that time
through FieldSpaceNN's `TimeEmbedder` (Time2Vec-style sin/cos at periods of 1,
7, 30.4375 and 365.25 days, the values in FieldSpaceNN's
`configs/embedding/default.yaml`, applied as scale/shift). Each timestep gets
its own valid time: the encoder uses the snapshot's, the processor window uses
each cached latent's plus the future slot's, and the decoder uses the time it
decodes for. The spacing of the future slot is measured from consecutive
snapshot times, not assumed.

The reservoir and the processor
-------------------------------
:class:`LatentReservoir` keeps the last ``n_history + rollout_steps - 1``
compressed states and their times, detached. The processor sees the
``n_history`` most recent ones stacked on FieldSpaceNN's time axis ``t``, plus
one future slot filled with a copy of the latest state (FieldSpaceNN's
``mask_ts_mode="repeat"`` convention for masked future timesteps) and stamped
with the future valid time. The attention blocks use
``token_len_time = n_history + 1``, so the whole window becomes one token per
coarse cell: attention runs over space only, while the projections have
separate weights per time position. The blocks start near-identity
(FieldSpaceNN's ~1e-12 gates), so the untrained prediction is persistence of
the latest latent.

What happens at ICON step t (``n_history = H``, ``rollout_steps = 1``)
--------------------------------------------------------------------
1. The reservoir holds z_{t-H} .. z_{t-1}, each compressed when its step
   arrived.
2. x_t arrives: it is the truth for the prediction made from z_{t-H} .. z_{t-1}.
   That prediction is recomputed here, with gradient. The weights have not
   changed since it was made and written to ICON at step t-1, so it is the
   same number.
3. Loss = MSE(prediction of x_t, x_t) + ``recon_weight`` * MSE(decode(encode(
   x_t)), x_t), then one optimizer step.
4. z_t = encode(x_t), from step 3's forward pass, is cached. The ring buffer
   drops z_{t-H}.
5. Predict x_{t+1} from z_{t-H+1} .. z_t and write it to ICON (plugin side).

With ``rollout_steps = n > 1``, step 2 instead rolls the processor n times
from the window ending at z_{t-n}.

Why the reconstruction term: cached latents are detached, so the prediction
loss cannot reach the encoder. Their graphs are gone, and the optimizer has
since changed the weights they were computed with in place. Without the
reconstruction loss the encoder would never train. With it, the encoder learns
from reconstruction as in the paper. A cached latent was produced by the
encoder as it was when it was cached, at most ``n_history + rollout_steps - 1``
optimizer steps ago. That staleness is the price of never keeping past *fine*
fields in memory.
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
from fieldspacenn.src.models.mg_transformer.mg_transformer import MG_Transformer
from fieldspacenn.src.modules.field_space.field_space_attention import (
    FieldSpaceAttentionConfig,
)
from fieldspacenn.src.modules.field_space.field_space_layer import FieldSpaceLayerConfig
from fieldspacenn.src.modules.grids.grid_utils import decode_zooms, encode_zooms

from fieldspace_online import _COARSE_ZOOM, _FINE_ZOOM, _MIN_ZOOM_CELLS, FieldSpaceSnapshot
from icon_mgrid_utils import LocalMGrid, pool_fine_to_coarse

# A latent (or a window of latents) in FieldSpaceNN's layout:
# {zoom: (b=1, v=1, t, n_zoom, d=nlev, f=latent_channels)}.
Latent = Dict[int, torch.Tensor]

_SECONDS_PER_DAY = 86400.0
# TimeEmbedder periods, in days (FieldSpaceNN configs/embedding/default.yaml).
_TIME_SCALES_DAYS = [1.0, 7.0, 30.4375, 365.25]


# ----------------------------------------------------------------------------
# Reservoir
# ----------------------------------------------------------------------------


@dataclass
class LatentWindow:
    latent: Latent  # {zoom: (1, 1, length, n, d, f)}, oldest first
    unix_seconds: List[float]  # valid time of each state, oldest first


class LatentReservoir:
    """Ring buffer of compressed states from previous timesteps, one
    ``{zoom: (1, 1, 1, n, d, f)}`` dict plus its valid time per timestep,
    oldest first.

    Everything is stored detached and copied, so no autograd graph or
    in-place mutation reaches across timesteps: the buffer holds data, never
    activations. Not checkpointed. After a restart it refills within
    ``capacity`` steps.
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
        oldest first, ending ``lag`` states before the most recent one
        (``lag=0``: ending at the most recent).

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


class FieldSpaceAEForecaster(nn.Module):
    """Encoder/decoder (`MG_AutoEncoder`) plus temporal processor
    (`MG_Transformer`) over one rank's compact two-zoom local patch, all
    time-conditioned. See the module docstring for the architecture.

    :meth:`forward` is the single training entry point (one call per
    backward, so DDP's reducer sees every parameter used exactly once per
    iteration); :meth:`encode`, :meth:`decode` and :meth:`rollout` are the
    building blocks, also used directly for inference. Times are unix seconds.
    """

    def __init__(
        self,
        mgrid: LocalMGrid,
        nlev: int,
        n_history: int,
        latent_channels: int = 4,
        n_blocks: int = 1,
        n_processor_blocks: int = 2,
        att_dim: int = 32,
        n_head_channels: int = 8,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.mgrid = mgrid
        self.nlev = int(nlev)
        self.n_history = int(n_history)
        self.latent_channels = int(latent_channels)

        mgrids = [
            {
                "coords": mgrid.coarse_coords,
                "adjc": mgrid.coarse_adjc,
                "adjc_mask": mgrid.coarse_adjc_mask,
                "zoom": _COARSE_ZOOM,
            },
            {
                "coords": mgrid.fine_coords,
                "adjc": mgrid.fine_adjc,
                "adjc_mask": mgrid.fine_adjc_mask,
                "zoom": _FINE_ZOOM,
            },
        ]
        both = [_COARSE_ZOOM, _FINE_ZOOM]
        latent = [_COARSE_ZOOM]

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
                token_zoom=_COARSE_ZOOM,
                q_zooms=zooms,
                kv_zooms=zooms,
                att_dim=att_dim,
                n_head_channels=n_head_channels,
                token_len_time=token_len_time,
                embed_confs=time_embed_confs,
            )

        encoder_blocks: Dict[str, Any] = {}
        for i in range(n_blocks):
            encoder_blocks[f"att_{i}"] = attention(both)
        # Paper eq 6: per coarse cell, [mean, 4 residual children] -> latent_channels.
        encoder_blocks["compress"] = FieldSpaceLayerConfig(
            in_zooms=both,
            target_zooms=latent,
            field_zoom=_COARSE_ZOOM,
            out_zooms=latent,
            target_features=self.latent_channels,
            type="mlp",
        )
        for i in range(n_blocks):
            encoder_blocks[f"latent_att_{i}"] = attention(latent)

        decoder_blocks: Dict[str, Any] = {}
        for i in range(n_blocks):
            decoder_blocks[f"latent_att_{i}"] = attention(latent)
        # Paper eq 10: the mirror, latent_channels -> [mean, 4 residual children].
        decoder_blocks["decompress"] = FieldSpaceLayerConfig(
            in_zooms=latent,
            target_zooms=both,
            field_zoom=_COARSE_ZOOM,
            out_zooms=both,
            target_features=1,
            type="mlp",
        )
        for i in range(n_blocks):
            decoder_blocks[f"att_{i}"] = attention(both)

        self.autoencoder = MG_AutoEncoder(
            mgrids=mgrids,
            in_zooms=both,
            encoder_block_configs=encoder_blocks,
            decoder_block_configs=decoder_blocks,
            in_features=1,
            n_head_channels=n_head_channels,
        )
        self.latent_zooms: List[int] = list(self.autoencoder.bottleneck_zooms)

        self.processor = MG_Transformer(
            mgrids=mgrids,
            block_configs={
                f"att_{i}": attention(self.latent_zooms, token_len_time=self.n_history + 1)
                for i in range(n_processor_blocks)
            },
            in_zooms=self.latent_zooms,
            in_features=self.latent_channels,
            n_head_channels=n_head_channels,
        )

        # Whole local patch as one patch. The past/future counts are only
        # compared *between* zooms by FieldSpaceNN's time-patch matching, so
        # they just have to be consistent within each call. They are set to
        # what each call actually holds: one timestep for encode/decode, and
        # n_history past + 1 future slot for the processor.
        self._sample_configs_step = {
            zoom: {"n_past_ts": 1, "n_future_ts": 0, "zoom_patch_sample": -1, "mask_n_last_ts": 0}
            for zoom in both
        }
        self._sample_configs_window = {
            zoom: {
                "n_past_ts": self.n_history, "n_future_ts": 1,
                "zoom_patch_sample": -1, "mask_n_last_ts": 1,
            }
            for zoom in both
        }

    def _time_emb(self, unix_seconds: Sequence[float]) -> Dict[str, Any]:
        """FieldSpaceNN embedding input for the given valid times (one per
        timestep on the ``t`` axis): ``{"TimeEmbedder": {zoom: (1, t) days}}``.

        Days since the unix epoch, as float32 like FieldSpaceNN's own dataset
        times: around 2e4 days that resolves about 2 minutes, well below
        ICON's time step. A fresh dict every call, because FieldSpaceNN
        blocks add zoom keys to it in place.
        """
        days = torch.tensor(
            [[float(s) / _SECONDS_PER_DAY for s in unix_seconds]],
            dtype=torch.float32,
            device=self.processor.zooms.device,
        )
        return {"TimeEmbedder": {_COARSE_ZOOM: days, _FINE_ZOOM: days}}

    def to_pyramid(self, x_fine_compact: torch.Tensor) -> Latent:
        """(4*n_coarse, nlev) -> {coarse: mean, fine: zero-mean residual} in
        (b, v, t, n, d, f) layout, via FieldSpaceNN's own `encode_zooms`."""
        n_coarse = self.mgrid.n_coarse
        coarse = pool_fine_to_coarse(x_fine_compact, n_coarse).reshape(1, 1, 1, n_coarse, self.nlev, 1)
        # reshape(), not view(): the compact tensor may be non-contiguous, and
        # encode_zooms subtracts in place, so the fine zoom must never alias
        # the caller's snapshot.
        fine = x_fine_compact.reshape(1, 1, 1, 4 * n_coarse, self.nlev, 1).clone()
        return encode_zooms({_COARSE_ZOOM: coarse, _FINE_ZOOM: fine}, self._sample_configs_step, None)

    def encode(self, x_fine_compact: torch.Tensor, unix_seconds: float) -> Latent:
        return self.autoencoder.ae_encode(
            [self.to_pyramid(x_fine_compact)],
            sample_configs=self._sample_configs_step,
            emb_groups=[self._time_emb([unix_seconds])],
        )[0]

    def decode(self, latent: Latent, unix_seconds: float) -> torch.Tensor:
        out = self.autoencoder.ae_decode(
            [dict(latent)],
            sample_configs=self._sample_configs_step,
            emb_groups=[self._time_emb([unix_seconds])],
            out_zoom=_FINE_ZOOM,
        )[0][_FINE_ZOOM]
        return out.reshape(self.mgrid.n_fine_kept, self.nlev)

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
    :class:`LatentReservoir`, performing online training on one rank's compact
    two-zoom local patch.

    Every call to :meth:`train_step` caches the snapshot's latent, and trains
    once the reservoir holds ``n_history + rollout_steps - 1`` earlier
    latents. Every rank pushes the same number of snapshots, so all ranks
    leave warm-up on the same step and call backward together, as DDP
    requires. As in `fieldspace_online.OnlineFieldSpaceTrainer`, a skip that
    happens on only some ranks (NaN input, too-small patch) breaks that
    lockstep. That risk is known and not handled here.
    """

    def __init__(
        self,
        nlev: int,
        mgrid: LocalMGrid,
        n_history: int = 4,
        rollout_steps: int = 1,
        latent_channels: int = 4,
        n_blocks: int = 1,
        n_processor_blocks: int = 2,
        recon_weight: float = 1.0,
        lr: float = 2e-4,
        att_dim: int = 32,
        n_head_channels: int = 8,
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

        # Same GridLayer lower bound as fieldspace_online (see _MIN_ZOOM_CELLS there).
        self._usable = mgrid.n_coarse >= _MIN_ZOOM_CELLS
        if not self._usable:
            self.model = None
            self.forward_model = None
            self.optimizer = None
            self.log_fn(
                f"[rank={rank}] FieldSpace AE trainer NOT built: n_coarse={mgrid.n_coarse} "
                f"< {_MIN_ZOOM_CELLS} — this rank will skip training/prediction every step."
            )
            return

        # Same default-device guard as fieldspace_online._build_model: FieldSpaceNN
        # creates device-less tensors inside GridLayer.__init__.
        with torch.device(self.device):
            self.model = FieldSpaceAEForecaster(
                mgrid,
                nlev=self.nlev,
                n_history=self.n_history,
                latent_channels=latent_channels,
                n_blocks=n_blocks,
                n_processor_blocks=n_processor_blocks,
                att_dim=att_dim,
                n_head_channels=n_head_channels,
                time_embed_dim=time_embed_dim,
            ).to(self.device)
        self.forward_model = self._wrap_ddp(use_ddp)
        self.optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=lr, weight_decay=0.0)

        n_params = sum(p.numel() for p in self.model.parameters())
        latent_numel = self.mgrid.n_coarse * self.nlev * latent_channels
        self.log_fn(
            f"[rank={rank}] FieldSpace AE trainer initialized: nlev={nlev}, "
            f"n_history={n_history}, rollout_steps={rollout_steps}, "
            f"latent_channels={latent_channels} (latent/native size="
            f"{latent_numel / (self.mgrid.n_fine_kept * self.nlev):.2f}), "
            f"reservoir_capacity={self.reservoir.capacity}, att_dim={att_dim}, "
            f"n_coarse={mgrid.n_coarse}, n_fine_kept={mgrid.n_fine_kept}, "
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

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self.mgrid.fine_compact_owned_mask.unsqueeze(-1)
        diff2 = (pred - target) ** 2 * mask
        return diff2.sum() / (mask.sum() * pred.shape[-1]).clamp(min=1)

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

        Loss = masked MSE(prediction, x) + ``recon_weight`` * masked MSE(
        reconstruction, x), over owned fine cells in complete quad-groups. The
        prediction is decoded from ``rollout_steps`` processor steps starting
        at the ``n_history`` cached latents that end ``rollout_steps`` steps
        before ``snapshot``.

        A NaN/Inf snapshot is neither trained on nor cached. The reservoir is
        cleared instead, to keep the window contiguous in time.
        """
        if not self._usable:
            self.log_fn("[trainer] model not built (n_coarse too small) — skipping step")
            return self._skip_result()
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
        # Same default-device guard as in __init__, for tensors FieldSpaceNN
        # creates during the forward pass.
        with torch.device(self.device):
            recon, pred, latent_now = self.forward_model(
                x, now, window.latent, window.unix_seconds, self.rollout_steps, self.step_seconds
            )
        loss_pred = self._masked_mse(pred, x)
        loss_recon = self._masked_mse(recon, x)
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

        Returns ``(4*n_coarse, nlev)`` normalized, or ``None`` while the
        reservoir holds fewer than ``n_history`` latents, before the step
        length is known (second snapshot), or if this rank's model was never
        built.
        """
        if not self._usable or self.step_seconds is None:
            return None
        window = self.reservoir.window(self.n_history)
        if window is None:
            return None
        self.model.eval()
        valid_seconds = window.unix_seconds[-1] + n_steps * self.step_seconds
        with torch.device(self.device):
            latent = self.model.rollout(window.latent, window.unix_seconds, n_steps, self.step_seconds)
            return self.model.decode(latent, valid_seconds)
