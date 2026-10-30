"""Online RECONSTRUCTOR + FORECASTER on one rank's nested ICON patch
(`.claude/schematic.pdf`).

Notation as in the schematic: x is the ICON field, c = E(x) its compressed
state (latent), ĉ a forecast, D the decoder. A bar, c̄ = Ē(x), marks the
frozen copy of the encoder (below). h are the forcings.

Levels
------
``levels`` are R2B levels, backbone first, e.g. ``[4, 6, 7, 8]`` on an R2B8
run; zoom ``z`` is ``level - levels[0]``. A field is split into the backbone
mean and one zero-mean residual per finer level with FieldSpaceNN's `to_zoom`
and `encode_zooms`, and summed back with `decode_zooms`. Both losses below are
FieldSpaceNN's `MGMultiLoss`, one MSE per level with weight 1 (as in
FieldSpaceNN's configs). On the field levels that is the whole-field MSE, so
each level counts by its share of the variance.

RECONSTRUCTOR
    x_t -> E(. | h_t) -> c_t -> D(. | h_t) -> x̂_t, an `MG_AutoEncoder` with
    the staged compression of the schematic: each residual level finer than
    ``latent_level`` is folded into the next coarser one. Trained with
    ``L_recon`` = per-level MSE(D(E(x_t)), x_t).

FORECASTER
    context c̄_{t-H} .. c̄_{t-1} -> MG transformer (. | h) -> ĉ_t, with
    H = ``n_context``. Trained with ``L_latent`` = per-latent-level
    MSE(ĉ_t, c̄_t): it forecasts the next compressed state directly, with no
    decoder in the loss, averaged over a batch of windows (latent replay,
    below). What is written to ICON is the increment through the frozen
    decoder, D̄(ĉ_{t+1}) - D̄(c̄_t).

Forcings
--------
The field x is the forecast variable only (e.g. ``uas``). The forcings h (e.g.
SST, sea ice, TOA incoming shortwave, orography) are not compressed: they
condition the blocks. Every attention block of both models embeds the
forcings of each cell at the block's finest zoom (:class:`CellForcingEmbedder`,
added to the time embedding) and modulates its normalized tokens with a
shift and scale computed from them (FieldSpaceNN's AdaLN ``shift_scale``).
Each token is one backbone cell, so it sees the forcings of all its cells at
that zoom: the R2B8 orography in the reconstructor's R2B8 blocks. So the
latent only has to code what the forcings do not explain, and the decoder has
the exact forcings. A zoom's forcing value is the mean over its cells.

- The reconstructor gets the columns ``reconstructor_forcings`` at time t,
  at every field zoom. The plugin leaves the insolation out of them: it is a
  clock, and a clock made the reconstructor's r8 fail at some hours of the
  day (job 28418236).
- The forecaster gets every forcing on its latent zoom, per time slot: those
  of each context step, and for the forecast slot those at the forecast time
  as far as they are known (the plugin: insolation at the forecast time, SST
  and sea ice of now). The replay buffer keeps each step's forecaster
  forcings with its latent.
- Both decodes of the written increment use the same forcings and time.

At sample step t (context full)
-------------------------------
1. c̄_t = Ē(x_t | h_t).
2. ``L_recon`` on x_t plus ``L_latent`` on the batch, one backward, one
   optimizer step (separate learning rates and gradient clipping per model),
   then the EMA update.
3. c̄_t goes into the replay buffer, where it joins the context and makes
   the live window replayable, and the plugin asks for the increment to t+1.
4. Returned for the log, per level: both losses and their references,
   ``state_ms`` (mean square of x_t, the loss of outputting zero, so
   1 - recon / state_ms is the fraction of variance explained) and
   ``latent_persist`` (mean square of c̄_t - c̄_{t-1} over the batch, the loss
   of persistence).

The two are trained independently: the forecaster's loss reaches only the
forecaster. The frozen encoder/decoder is an exponential moving average of the
reconstructor (`torch.optim.swa_utils.AveragedModel`). It encodes every
latent the forecaster sees or is scored against, so the forecaster works in a
latent space that moves slowly and consistently while the reconstructor keeps
learning.

Latent replay
-------------
Against forgetting earlier weather while training on a correlated stream. A
window is one forecaster training sample: the context c̄_{t-H} .. c̄_{t-1},
the target c̄_t, their forecaster forcings and their valid times. The replay
buffer keeps the frozen latents and forcings of the last
``n_context + replay_size`` sample steps, each once: the newest ``n_context``
are the live context, and every earlier step is the target of one stored
window, so ``replay_size`` windows can be replayed. Windows are assembled when
drawn. Every update draws ``replay_batch`` of them at random, and the
forecaster is trained on these plus the new (live) window as one batch.
Stored latents are not re-encoded: the buffer length bounds how much older the
encoder that made them is, and windows from the first ``replay_warmup`` sample
steps, encoded before the reconstructor had settled, are never replayed.
"""

from collections import deque
from typing import Dict, List, NamedTuple, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from fieldspacenn.src.models.mg_autoencoder.mg_autoencoder import MG_AutoEncoder
from fieldspacenn.src.models.mg_transformer.mg_base_model import create_encoder_decoder_block
from fieldspacenn.src.modules.embedding import embedder as fs_embedder
from fieldspacenn.src.modules.field_space.field_space_attention import FieldSpaceAttentionConfig
from fieldspacenn.src.modules.field_space.field_space_layer import FieldSpaceLayerConfig
from fieldspacenn.src.modules.grids.grid_utils import decode_zooms, encode_zooms, to_zoom
from fieldspacenn.src.utils.losses import MGMultiLoss

# {zoom: (b, v=1, t, n_zoom, d=nlev, f)}, FieldSpaceNN's layout.
Levels = Dict[int, torch.Tensor]
# {zoom: (b, t, n_zoom, n_forcings)}, the forcings at a zoom; {} without forcings.
Forcings = Dict[int, torch.Tensor]

# TimeEmbedder periods in days, as in FieldSpaceNN's configs/embedding/default.yaml.
# FieldSpaceNN's configs also have a 7-day period. It is left out: the atmosphere has no weekly
# cycle, and it let the model recall what it learned at the same weekday phase (job 28401149).
_TIME_SCALES_DAYS = [1.0, 30.4375, 365.25]


class CellForcingEmbedder(fs_embedder.StaticVariableEmbedder):
    """FieldSpaceNN's `StaticVariableEmbedder` (an MLP of each cell's values)
    without its neighbourhood: that gathers neighbours as on a whole grid and
    rejects a patch, whose edge cells have fewer neighbours inside it. The
    token already holds all cells of a backbone cell. Input: the forcings at
    the block's zoom, (b, t, n_zoom, n_forcings)."""

    def __init__(self, name, in_channels, embed_dim, zoom, **kwargs):
        super().__init__(name, in_channels, embed_dim, zoom)

    def forward(self, forcings: Forcings, **kwargs) -> torch.Tensor:
        return self.embedding_fn(forcings[self.zoom])


# FieldSpaceNN builds embedders by class name from its embedder module.
fs_embedder.CellForcingEmbedder = CellForcingEmbedder


def _sample_configs(n_past: int, n_future: int, zooms: Sequence[int]) -> Dict[int, Dict]:
    """Whole patch as one sample, ``n_past`` + ``n_future`` timesteps."""
    conf = {"n_past_ts": n_past, "n_future_ts": n_future, "zoom_patch_sample": -1, "mask_n_last_ts": n_future}
    return {zoom: dict(conf) for zoom in zooms}


class Window(NamedTuple):
    """One forecaster training sample, batch size 1."""

    context: Levels  # frozen latents c̄_{t-H} .. c̄_{t-1}
    target: Levels  # frozen latent c̄_t
    forcings: Forcings  # forecaster forcings of the context steps, then of the target step
    seconds: List[float]  # valid times of the context, then of the target


def stack_windows(windows: Sequence[Window]) -> Window:
    """Windows concatenated on the batch axis; ``seconds`` becomes one row per
    window."""
    return Window(*(
        {z: torch.cat([getattr(w, field)[z] for w in windows]) for z in getattr(windows[0], field)}
        for field in ("context", "target", "forcings")
    ), [w.seconds for w in windows])


class ReplayBuffer:
    """Frozen latents, forecaster forcings and valid times of the last
    ``n_context + size`` sample steps, each stored once. The newest
    ``n_context`` are the live context; every earlier step is the target of a
    stored window. :meth:`sample` draws up to ``batch`` stored windows at
    random, without replacement. ``size=0`` turns replay off. Windows that
    touch the first ``warmup`` steps are never drawn: their latents come from a
    reconstructor that had barely trained."""

    def __init__(self, n_context: int, size: int, batch: int, generator: torch.Generator, warmup: int = 0):
        self.n_context, self.warmup = n_context, warmup
        self.steps = deque(maxlen=n_context + size)  # (seconds, latent, forcings), oldest first
        self.n_appended = 0
        self.batch, self._rng = batch, generator

    def append(self, seconds: float, latent: Levels, forcings: Forcings) -> None:
        self.steps.append((seconds, latent, forcings))
        self.n_appended += 1

    @property
    def context_full(self) -> bool:
        return len(self.steps) >= self.n_context

    def context(self, end: Optional[int] = None):
        """Latents and forcings of the ``n_context`` steps before step ``end``
        (default: the newest ones), stacked on the time axis, and their valid
        times."""
        end = len(self.steps) if end is None else end
        seconds, latents, forcings = zip(*(self.steps[i] for i in range(end - self.n_context, end)))
        latents = {z: torch.cat([lv[z] for lv in latents], dim=2) for z in latents[0]}
        forcings = {z: torch.cat([f[z] for f in forcings], dim=1) for z in forcings[0]}
        return latents, forcings, list(seconds)

    def window(self, end: int) -> Window:
        """The window whose target is step ``end``."""
        context, forcings, seconds = self.context(end)
        target_seconds, target, target_forcings = self.steps[end]
        forcings = {z: torch.cat([f, target_forcings[z]], dim=1) for z, f in forcings.items()}
        return Window(context, target, forcings, seconds + [target_seconds])

    def sample(self) -> List[Window]:
        # Earliest drawable target: its whole context comes after the warm-up steps.
        n_dropped = self.n_appended - len(self.steps)  # steps already pushed out of the deque
        first = max(self.n_context, self.warmup + self.n_context - n_dropped)
        n_windows = len(self.steps) - first
        n = min(self.batch, n_windows)
        if n <= 0:
            return []
        ends = torch.randperm(n_windows, generator=self._rng, device="cpu")[:n] + first
        return [self.window(end) for end in ends.tolist()]


class _BlockConfigs:
    """FieldSpaceNN block configs of one model: attention tokens on the
    backbone, conditioned on time and ``n_forcings`` forcings, ``ext`` blocks
    (the legacy blocks reject levels with different channel counts)."""

    def __init__(self, att_dim, n_head_channels, hidden_dim, embed_dim, n_forcings):
        self.att_dim, self.n_head_channels, self.hidden_dim = att_dim, n_head_channels, hidden_dim
        self.embed_dim, self.n_forcings = embed_dim, n_forcings

    def embedding(self, zoom: int) -> Dict:
        """Time embedding plus, with forcings, their embedding at ``zoom``
        (summed: one ``embed_dim`` for both)."""
        confs = {
            "TimeEmbedder": {
                "in_channels": 1,
                "embed_dim": self.embed_dim,
                "time_scales": _TIME_SCALES_DAYS,
                "time_min": 0.0,
                "time_max": 1.0,
                "use_linear": False,
                # FieldSpaceNN's shared-embedder cache is a process-wide singleton.
                "shared": False,
            },
            "CellForcingEmbedder": {"in_channels": self.n_forcings, "embed_dim": self.embed_dim, "shared": False},
        }
        names = list(confs)[: 1 + bool(self.n_forcings)]
        return {"embed_names": names, "embed_mode": "sum", "embed_confs": confs, "input_zoom": zoom}

    def attention(self, zooms, token_len_time=1):
        return FieldSpaceAttentionConfig(
            token_zoom=0,
            q_zooms=list(zooms),
            kv_zooms=list(zooms),
            att_dim=self.att_dim,
            n_head_channels=self.n_head_channels,
            token_len_time=token_len_time,
            embed_confs=self.embedding(max(zooms)),
            block_type="ext",
        )

    def field_layer(self, in_zooms, target_zooms, field_zoom, out_zooms, target_features):
        return FieldSpaceLayerConfig(
            in_zooms=in_zooms,
            target_zooms=target_zooms,
            field_zoom=field_zoom,
            out_zooms=out_zooms,
            target_features=target_features,
            type="mlp",
            block_type="ext",
            hidden_dim=self.hidden_dim,
        )


def autoencoder_configs(zooms: List[int], latent_zoom: int, channels: int, n_blocks: int, configs: _BlockConfigs):
    """Encoder and decoder block configs of the staged compression, and the
    channels of every latent level. The finest residual level is folded into
    the next coarser one, stage by stage, down to ``latent_zoom`` (``channels``
    per latent cell), with ``n_blocks`` attention blocks before and after each
    stage."""
    stages = [(fine, parent) for parent, fine in zip(zooms[1:], zooms[2:]) if fine > latent_zoom][::-1]
    zooms = list(zooms)
    encoder = {f"att_in_{i}": configs.attention(zooms) for i in range(n_blocks)}
    for s, (fine, parent) in enumerate(stages):
        zooms.remove(fine)
        encoder[f"compress_{s}"] = configs.field_layer([parent, fine], [parent], parent, list(zooms), channels)
        encoder.update({f"att_{s}_{i}": configs.attention(zooms) for i in range(n_blocks)})
    latent_features = {z: channels if z in dict(stages).values() else 1 for z in zooms}
    decoder = {f"latent_att_{i}": configs.attention(zooms) for i in range(n_blocks)}
    for s, (fine, parent) in reversed(list(enumerate(stages))):
        zooms = sorted(zooms + [fine])
        fine_channels = channels if s > 0 else 1  # fine was itself a parent in the stage before
        decoder[f"decompress_{s}"] = configs.field_layer(
            [parent], [parent, fine], parent, list(zooms), {parent: 1, fine: fine_channels}
        )
        decoder.update({f"att_{s}_{i}": configs.attention(zooms) for i in range(n_blocks)})
    return encoder, decoder, latent_features


class Reconstructor(nn.Module):
    """Encoder/decoder between the field's levels and its latent."""

    def __init__(self, mgrids, zooms, latent_zoom, channels, n_blocks, configs: _BlockConfigs):
        super().__init__()
        encoder, decoder, self.latent_features = autoencoder_configs(zooms, latent_zoom, channels, n_blocks, configs)
        self.ae = MG_AutoEncoder(
            mgrids=mgrids,
            in_zooms=zooms,
            encoder_block_configs=encoder,
            decoder_block_configs=decoder,
            n_head_channels=configs.n_head_channels,
        )
        self.sample_configs = _sample_configs(1, 0, range(len(mgrids)))

    # FieldSpaceNN blocks write their outputs into the input dict: pass copies.
    def encode(self, levels: Levels, emb) -> Levels:
        return self.ae.ae_encode([dict(levels)], sample_configs=self.sample_configs, emb_groups=[emb])[0]

    def decode(self, latent: Levels, emb) -> Levels:
        return self.ae.ae_decode([dict(latent)], sample_configs=self.sample_configs, emb_groups=[emb])[0]


class Forecaster(nn.Module):
    """MG transformer over the latent levels: the ``n_context`` context
    latents plus one future slot (a copy of the latest, FieldSpaceNN's
    ``repeat`` convention) form one token per backbone cell; the future slot is
    returned. The gates start near zero, so the untrained forecast is
    persistence."""

    def __init__(self, grid_layers, latent_features: Dict[int, int], n_context, n_blocks, configs: _BlockConfigs):
        super().__init__()
        zooms = sorted(latent_features)
        self.blocks = nn.ModuleList(
            create_encoder_decoder_block(
                configs.attention(zooms, token_len_time=n_context + 1),
                zooms,
                [latent_features[z] for z in zooms],
                [1],
                grid_layers=grid_layers,
                n_head_channels=configs.n_head_channels,
            )
            for _ in range(n_blocks)
        )
        self.sample_configs = _sample_configs(n_context, 1, range(len(grid_layers)))

    def forward(self, context: Levels, emb) -> Levels:
        x = [{z: torch.cat([c, c[:, :, -1:]], dim=2) for z, c in context.items()}]  # append the future slot
        for block in self.blocks:
            x = block(x, sample_configs=self.sample_configs, emb_groups=[emb])
        return {z: t[:, :, -1:] for z, t in x[0].items()}


class ReconstructorForecaster(nn.Module):
    """Both models plus the frozen (EMA) copy of the reconstructor."""

    def __init__(
        self,
        mgrids,
        levels: Sequence[int],
        latent_level: int,
        nlev: int,
        n_context: int = 4,
        n_forcings: int = 0,
        reconstructor_forcings: Sequence[int] = (),
        latent_channels: int = 4,
        n_blocks: int = 2,
        n_forecaster_blocks: int = 2,
        att_dim: int = 128,
        n_head_channels: int = 8,
        hidden_dim: int = 64,
        embed_dim: int = 8,
        ema_decay: float = 0.99,
        reconstructor_time: bool = True,
    ):
        """``n_forcings`` forcing columns condition the forecaster, the columns
        ``reconstructor_forcings`` of them the reconstructor. ``embed_dim`` is
        the size of the time embedding and of each cell's forcing embedding.
        ``reconstructor_time=False`` gives the reconstructor a constant time:
        it compresses and rebuilds a field without the clock, while the
        forecaster still gets the model time."""
        super().__init__()
        self.reconstructor_time = reconstructor_time
        self.reconstructor_forcings = list(reconstructor_forcings)
        self.zooms = [level - levels[0] for level in levels]
        self.fine_zoom = self.zooms[-1]
        self.nlev = nlev
        self.all_zooms = list(range(len(mgrids)))
        self.sample_configs = _sample_configs(1, 0, self.all_zooms)
        blocks = dict(att_dim=att_dim, n_head_channels=n_head_channels, hidden_dim=hidden_dim, embed_dim=embed_dim)

        self.reconstructor = Reconstructor(
            mgrids,
            self.zooms,
            latent_level - levels[0],
            latent_channels,
            n_blocks,
            _BlockConfigs(**blocks, n_forcings=len(self.reconstructor_forcings)),
        )
        self.forecaster = Forecaster(
            self.reconstructor.ae.grid_layers,
            self.reconstructor.latent_features,
            n_context,
            n_forecaster_blocks,
            _BlockConfigs(**blocks, n_forcings=n_forcings),
        )
        self.forecast_zoom = max(self.reconstructor.latent_features)  # the forecaster's forcings
        self.frozen = AveragedModel(self.reconstructor, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        self.frozen.requires_grad_(False)

    def emb(self, times: Sequence[Sequence[float]], forcings: Forcings):
        """FieldSpaceNN's embedding inputs, on the default device (the trainer
        sets it): one row of valid times (unix seconds) along the time axis per
        batch sample, and the forcings unless there are none. A new dict every
        call: FieldSpaceNN blocks add keys to it."""
        days = torch.tensor([[s / 86400.0 for s in row] for row in times])
        emb = {"TimeEmbedder": {zoom: days for zoom in self.all_zooms}}
        if forcings:
            emb["CellForcingEmbedder"] = forcings
        return emb

    def to_levels(self, x: torch.Tensor) -> Levels:
        """(n_fine, nlev) -> backbone mean and zero-mean residuals per level."""
        x = x.reshape(1, 1, 1, -1, self.nlev, 1).clone()  # encode_zooms works in place
        return encode_zooms({z: to_zoom(x, self.fine_zoom, z)[0] for z in self.zooms}, self.sample_configs, None)

    def to_field(self, levels: Levels) -> torch.Tensor:
        """Inverse of :meth:`to_levels`."""
        return decode_zooms(dict(levels), self.sample_configs, self.fine_zoom)[self.fine_zoom].reshape(-1, self.nlev)

    def forcing_levels(self, forcings: torch.Tensor) -> Forcings:
        """(n_fine, n_forcings) -> their means at every field zoom, {} without
        forcings."""
        if not forcings.shape[1]:
            return {}
        h = forcings[None, None, :, None]  # (1, t=1, n_fine, d=1, n_forcings), to_zoom's layout
        return {z: to_zoom(h, self.fine_zoom, z)[0][:, :, :, 0] for z in self.zooms}

    def reconstructor_emb(self, seconds: float, forcings: Forcings):
        """Embedding inputs of the reconstructor (live or frozen) for a field at
        ``seconds``: its time, or a constant without ``reconstructor_time``,
        and its forcings."""
        columns = self.reconstructor_forcings
        own = {z: f[..., columns] for z, f in forcings.items()} if columns else {}
        return self.emb([[seconds if self.reconstructor_time else 0.0]], own)

    def encode_frozen(self, levels: Levels, seconds: float, forcings: Forcings) -> Levels:
        return self.frozen.module.encode(levels, self.reconstructor_emb(seconds, forcings))

    def decode_frozen(self, latent: Levels, seconds: float, forcings: Forcings) -> Levels:
        return self.frozen.module.decode(latent, self.reconstructor_emb(seconds, forcings))

    def forecast(self, context: Levels, forcings: Forcings, times: Sequence[Sequence[float]]) -> Levels:
        """Latent forecast for a batch of contexts. ``forcings`` and ``times``
        cover the context steps, then the forecast; ``times`` has one row per
        sample."""
        return self.forecaster(context, self.emb(times, forcings))

    def forward(self, levels: Levels, seconds: float, forcings: Forcings, batch: Window):
        """Training pass: the reconstruction of x_t and the latent forecasts of
        a batch of windows. One call, so DDP sees every trainable parameter
        once."""
        latent = self.reconstructor.encode(levels, self.reconstructor_emb(seconds, forcings))
        recon = self.reconstructor.decode(latent, self.reconstructor_emb(seconds, forcings))
        return recon, self.forecast(batch.context, batch.forcings, batch.seconds)


class OnlineReconstructorForecaster:
    """Online training of :class:`ReconstructorForecaster` on one rank's patch,
    one call to :meth:`train_step` per sample step. See the module docstring."""

    def __init__(
        self,
        mgrids,
        levels: Sequence[int],
        latent_level: int,
        nlev: int,
        n_context: int = 4,
        lr_recon: float = 2e-4,
        lr_forecast: float = 2e-4,
        replay_size: int = 0,
        replay_batch: int = 10,
        replay_warmup: int = 0,
        grad_clip: float = 1.0,
        device: torch.device = torch.device("cuda", 0),
        **model_kwargs,
    ):
        self.device, self.grad_clip = device, grad_clip
        # FieldSpaceNN creates device-less tensors in GridLayer and in forward
        # passes; the grids must be on the same device as those.
        mgrids = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in grid.items()} for grid in mgrids]
        with torch.device(device):
            self.model = ReconstructorForecaster(mgrids, levels, latent_level, nlev, n_context, **model_kwargs)
        self.ddp_model = (
            DDP(self.model, broadcast_buffers=False)  # grid buffers differ: patch sizes differ
            if dist.is_initialized() and dist.get_world_size() > 1
            else self.model
        )
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.reconstructor.parameters(), "lr": lr_recon},
                {"params": self.model.forecaster.parameters(), "lr": lr_forecast},
            ]
        )
        self.level_loss = MGMultiLoss({str(z): {"MSE_loss": 1.0} for z in self.model.zooms})
        rng = torch.Generator().manual_seed(dist.get_rank() if dist.is_initialized() else 0)
        # Also holds the context. replay_warmup: sample steps after the dry run whose windows are never replayed.
        self.replay_buffer = ReplayBuffer(n_context, replay_size, replay_batch, rng, warmup=replay_warmup)

    def _loss(self, output: Levels, target: Levels):
        """Plain MGMultiLoss: the trained loss (sum of the per-level MSEs =
        whole-field MSE) and, for the log, the per-level MSE."""
        total, parts = self.level_loss(output, target)
        return total, {z: float(parts[f"group0_level{z}_var0_MSE_loss"]) for z in target}

    @staticmethod
    def _mean_square(levels: Levels) -> Dict[int, float]:
        """Per-level mean square: the MSE of predicting zero for ``levels``."""
        return {z: float(t.square().mean()) for z, t in levels.items()}

    def _forecaster_forcings(self, forcings: Forcings) -> Forcings:
        return {z: f for z, f in forcings.items() if z == self.model.forecast_zoom}

    def train_step(self, x: torch.Tensor, forcings: torch.Tensor, seconds: float) -> Dict:
        """Train on the normalized patch ``x`` (n_fine, nlev) with the
        normalized forcings (n_fine, n_forcings), both valid at ``seconds``,
        once the context is full, then add its frozen latent and forcings to
        the replay buffer, where they join the context. Returns the per-level
        losses for the log, empty while the context fills."""
        stats = {}
        self.model.train()
        with torch.device(self.device):
            levels, forcings = self.model.to_levels(x), self.model.forcing_levels(forcings)
            with torch.no_grad():
                latent = self.model.encode_frozen(levels, seconds, forcings)
            if self.replay_buffer.context_full:
                stats = self._update(levels, forcings, latent, seconds)
            self.replay_buffer.append(seconds, latent, self._forecaster_forcings(forcings))
        return stats

    def _update(self, levels: Levels, forcings: Forcings, latent: Levels, seconds: float) -> Dict:
        # The live window ends at c̄_t; the batch adds windows drawn from the replay buffer.
        context, context_forcings, context_seconds = self.replay_buffer.context()
        live_forcings = {z: torch.cat([f, forcings[z]], dim=1) for z, f in context_forcings.items()}
        live = Window(context, latent, live_forcings, context_seconds + [seconds])
        batch = stack_windows([live] + self.replay_buffer.sample())

        recon, forecast = self.ddp_model(levels, seconds, forcings, batch)
        loss_recon, recon_parts = self._loss(recon, levels)
        loss_latent, latent_parts = self._loss(forecast, batch.target)

        self.optimizer.zero_grad(set_to_none=True)
        (loss_recon + loss_latent).backward()
        # Gradients are averaged over ranks, so all ranks take the same branch.
        norms = [nn.utils.clip_grad_norm_(group["params"], self.grad_clip) for group in self.optimizer.param_groups]
        if all(torch.isfinite(norm) for norm in norms):
            self.optimizer.step()
            self.model.frozen.update_parameters(self.model.reconstructor)

        # Each loss per level, with the loss of outputting zero (x_t) or of persistence (c̄_{t-1} for c̄_t).
        persistence = {z: batch.target[z] - batch.context[z][:, :, -1:] for z in batch.target}
        return {
            "recon": recon_parts,
            "state_ms": self._mean_square(levels),
            "latent_mse": latent_parts,
            "latent_persist": self._mean_square(persistence),
        }

    @torch.no_grad()
    def predict_increment(self, seconds: float, forcings: torch.Tensor):
        """Change from the latest context state to ``seconds``,
        D̄(ĉ) - D̄(c̄_latest), both through the frozen decoder: (n_fine, nlev)
        normalized, or ``None`` while the context fills. ``forcings``
        (n_fine, n_forcings, normalized) are those at ``seconds`` as far as
        they are known; they condition the forecast slot and both decodes.
        Both decodes also use the time of c̄_latest, so any time dependence of
        the decoder cancels (job 28418236: decoding ĉ at its own valid time let
        it leak into r8)."""
        if not self.replay_buffer.context_full:
            return None
        self.model.eval()
        context, context_forcings, context_seconds = self.replay_buffer.context()
        last_seconds, last_latent, _ = self.replay_buffer.steps[-1]
        with torch.device(self.device):
            forcings = self.model.forcing_levels(forcings)
            slots = {z: torch.cat([f, forcings[z]], dim=1) for z, f in context_forcings.items()}
            forecast = self.model.forecast(context, slots, [context_seconds + [seconds]])
            forecast = self.model.decode_frozen(forecast, last_seconds, forcings)
            last = self.model.decode_frozen(last_latent, last_seconds, forcings)
            return self.model.to_field({z: forecast[z] - last[z] for z in forecast})
