"""Online RECONSTRUCTOR + FORECASTER on one rank's nested ICON patch
(`.claude/schematic.pdf`).

Notation as in the schematic: x is the ICON field, c = E(x) its compressed
state (latent), ĉ a forecast, D the decoder. A bar, c̄ = Ē(x), marks the
frozen copy of the encoder (below).

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
    x_t -> E -> c_t -> D -> x̂_t, an `MG_AutoEncoder` with the staged
    compression of the schematic: each residual level finer than
    ``latent_level`` is folded into the next coarser one. Trained with
    ``L_recon`` = per-level MSE(D(E(x_t)), x_t).

FORECASTER
    context c̄_{t-H} .. c̄_{t-1} -> MG transformer -> ĉ_t, with
    H = ``n_context``. Trained with ``L_latent`` = per-latent-level
    MSE(ĉ_t, c̄_t): it forecasts the next compressed state directly, with no
    decoder in the loss, averaged over a batch of windows (latent replay,
    below). What is written to ICON is the increment through the frozen
    decoder, D̄(ĉ_{t+1}) - D̄(c̄_t).

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
the target c̄_t and their valid times. The replay buffer keeps the last
``replay_size`` windows; every update draws ``replay_batch`` of them at random,
and the forecaster is trained on these plus the new (live) window as one batch.
Stored latents are not re-encoded: the buffer length bounds how much older the
encoder that made them is.

At sample step t (context full)
-------------------------------
1. c̄_t = Ē(x_t).
2. ``L_recon`` on x_t plus ``L_latent`` on the batch, one backward, one
   optimizer step (separate learning rates and gradient clipping per model),
   then the EMA update.
3. The live window goes into the replay buffer, c̄_t joins the context, and
   the plugin asks for the increment to t+1.
4. Returned for the log, per level: both losses and their references,
   ``state_ms`` (mean square of x_t, the loss of outputting zero) and
   ``latent_persist`` (mean square of c̄_t - c̄_{t-1} over the batch, the loss
   of persistence).
"""

from collections import deque
from typing import Dict, List, NamedTuple, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from fieldspacenn.src.models.mg_autoencoder.mg_autoencoder import MG_AutoEncoder
from fieldspacenn.src.models.mg_transformer.mg_base_model import create_encoder_decoder_block
from fieldspacenn.src.modules.field_space.field_space_attention import FieldSpaceAttentionConfig
from fieldspacenn.src.modules.field_space.field_space_layer import FieldSpaceLayerConfig
from fieldspacenn.src.modules.grids.grid_utils import decode_zooms, encode_zooms, to_zoom
from fieldspacenn.src.utils.losses import MGMultiLoss

# {zoom: (b, v=1, t, n_zoom, d=nlev, f)}, FieldSpaceNN's layout.
Levels = Dict[int, torch.Tensor]

# TimeEmbedder periods in days, as in FieldSpaceNN's configs/embedding/default.yaml.
_TIME_SCALES_DAYS = [1.0, 7.0, 30.4375, 365.25]


def _sample_configs(n_past: int, n_future: int, zooms: Sequence[int]) -> Dict[int, Dict]:
    """Whole patch as one sample, ``n_past`` + ``n_future`` timesteps."""
    conf = {"n_past_ts": n_past, "n_future_ts": n_future, "zoom_patch_sample": -1, "mask_n_last_ts": n_future}
    return {zoom: dict(conf) for zoom in zooms}


class Window(NamedTuple):
    """One forecaster training sample, batch size 1."""

    context: Levels  # frozen latents c̄_{t-H} .. c̄_{t-1}
    target: Levels  # frozen latent c̄_t
    seconds: List[float]  # valid times of the context, then of the target


def stack_windows(windows: Sequence[Window]) -> Window:
    """Windows concatenated on the batch axis; ``seconds`` becomes one row per
    window."""
    return Window(
        {z: torch.cat([w.context[z] for w in windows]) for z in windows[0].context},
        {z: torch.cat([w.target[z] for w in windows]) for z in windows[0].target},
        [w.seconds for w in windows],
    )


class ReplayBuffer:
    """The last ``size`` windows; :meth:`sample` draws up to ``batch`` of them
    at random, without replacement. ``size=0`` turns replay off."""

    def __init__(self, size: int, batch: int, generator: torch.Generator):
        self.windows = deque(maxlen=size)
        self.batch, self._rng = batch, generator

    def add(self, window: Window) -> None:
        self.windows.append(window)

    def sample(self) -> List[Window]:
        n = min(self.batch, len(self.windows))
        if n == 0:
            return []
        picks = torch.randperm(len(self.windows), generator=self._rng, device="cpu")[:n]
        return [self.windows[i] for i in picks.tolist()]


class _BlockConfigs:
    """FieldSpaceNN block configs shared by all models: attention tokens on
    the backbone, time conditioning, ``ext`` blocks (the legacy blocks reject
    levels with different channel counts)."""

    def __init__(self, att_dim, n_head_channels, hidden_dim, time_embed_dim):
        self.att_dim, self.n_head_channels, self.hidden_dim = att_dim, n_head_channels, hidden_dim
        self.time = {
            "embed_names": ["TimeEmbedder"],
            "embed_mode": "sum",
            "embed_confs": {
                "TimeEmbedder": {
                    "in_channels": 1,
                    "embed_dim": time_embed_dim,
                    "time_scales": _TIME_SCALES_DAYS,
                    "time_min": 0.0,
                    "time_max": 1.0,
                    "use_linear": False,
                    # FieldSpaceNN's shared-embedder cache is a process-wide singleton.
                    "shared": False,
                }
            },
        }

    def attention(self, zooms, token_len_time=1):
        return FieldSpaceAttentionConfig(
            token_zoom=0,
            q_zooms=list(zooms),
            kv_zooms=list(zooms),
            att_dim=self.att_dim,
            n_head_channels=self.n_head_channels,
            token_len_time=token_len_time,
            embed_confs=self.time,
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
    """Encoder/decoder between a field's levels and its latent."""

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
    def encode(self, levels: Levels, time_emb) -> Levels:
        return self.ae.ae_encode([dict(levels)], sample_configs=self.sample_configs, emb_groups=[time_emb])[0]

    def decode(self, latent: Levels, time_emb) -> Levels:
        return self.ae.ae_decode([dict(latent)], sample_configs=self.sample_configs, emb_groups=[time_emb])[0]


class Forecaster(nn.Module):
    """MG transformer over the latent levels: the ``n_context`` context
    latents plus one future slot (a copy of the latest, FieldSpaceNN's
    ``repeat`` convention) form one token per backbone cell; the future slot
    is returned. The gates start near zero, so the untrained forecast is
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

    def forward(self, context: Levels, time_emb) -> Levels:
        x = [{z: torch.cat([c, c[:, :, -1:]], dim=2) for z, c in context.items()}]  # append the future slot
        for block in self.blocks:
            x = block(x, sample_configs=self.sample_configs, emb_groups=[time_emb])
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
        latent_channels: int = 4,
        n_blocks: int = 2,
        n_forecaster_blocks: int = 2,
        att_dim: int = 128,
        n_head_channels: int = 8,
        hidden_dim: int = 64,
        time_embed_dim: int = 64,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.zooms = [level - levels[0] for level in levels]
        self.fine_zoom = self.zooms[-1]
        self.nlev = nlev
        self.all_zooms = list(range(len(mgrids)))
        self.sample_configs = _sample_configs(1, 0, self.all_zooms)
        configs = _BlockConfigs(att_dim, n_head_channels, hidden_dim, time_embed_dim)

        self.reconstructor = Reconstructor(
            mgrids, self.zooms, latent_level - levels[0], latent_channels, n_blocks, configs
        )
        self.forecaster = Forecaster(
            self.reconstructor.ae.grid_layers, self.reconstructor.latent_features, n_context, n_forecaster_blocks, configs
        )
        self.frozen = AveragedModel(self.reconstructor, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        self.frozen.requires_grad_(False)

    def time_emb(self, times: Sequence[Sequence[float]]):
        """Time embedding input: one row of valid times (unix seconds) along
        the time axis per batch sample, on the default device (the trainer
        sets it). A new dict every call: FieldSpaceNN blocks add keys to it."""
        days = torch.tensor([[s / 86400.0 for s in row] for row in times])
        return {"TimeEmbedder": {zoom: days for zoom in self.all_zooms}}

    def to_levels(self, x: torch.Tensor) -> Levels:
        """(n_fine, nlev) -> backbone mean and zero-mean residuals per level."""
        x = x.reshape(1, 1, 1, -1, self.nlev, 1).clone()  # encode_zooms subtracts in place
        return encode_zooms({z: to_zoom(x, self.fine_zoom, z)[0] for z in self.zooms}, self.sample_configs, None)

    def to_field(self, levels: Levels) -> torch.Tensor:
        """Inverse of :meth:`to_levels`."""
        return decode_zooms(dict(levels), self.sample_configs, self.fine_zoom)[self.fine_zoom].reshape(-1, self.nlev)

    def encode_frozen(self, levels: Levels, seconds: float) -> Levels:
        return self.frozen.module.encode(levels, self.time_emb([[seconds]]))

    def decode_frozen(self, latent: Levels, seconds: float) -> Levels:
        return self.frozen.module.decode(latent, self.time_emb([[seconds]]))

    def forecast(self, context: Levels, times: Sequence[Sequence[float]]) -> Levels:
        """Latent forecast of a batch of contexts; ``times`` has one row per
        sample: the context's valid times, then the forecast's."""
        return self.forecaster(context, self.time_emb(times))

    def forward(self, levels: Levels, seconds: float, context: Levels, times: Sequence[Sequence[float]]):
        """Training pass: the reconstruction of x_t and the latent forecasts of
        a batch of windows. One call, so DDP sees every trainable parameter
        once."""
        latent = self.reconstructor.encode(levels, self.time_emb([[seconds]]))
        recon = self.reconstructor.decode(latent, self.time_emb([[seconds]]))
        return recon, self.forecast(context, times)


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
        self.context = deque(maxlen=n_context)  # (seconds, frozen latent) of the latest sample steps
        rng = torch.Generator().manual_seed(dist.get_rank() if dist.is_initialized() else 0)
        self.replay_buffer = ReplayBuffer(replay_size, replay_batch, rng)

    def _loss(self, output: Levels, target: Levels):
        """Plain MGMultiLoss: the trained loss (sum of the per-level MSEs =
        whole-field MSE) and its per-level parts, which are what is logged."""
        total, parts = self.level_loss(output, target)
        return total, {z: float(parts[f"level{z}_MSE_loss"]) for z in target}

    @staticmethod
    def _mean_square(levels: Levels) -> Dict[int, float]:
        """Per-level mean square: the MSE of predicting zero for ``levels``."""
        return {z: float(t.square().mean()) for z, t in levels.items()}

    def _stacked_context(self):
        """The context latents stacked on the time axis, oldest first, and
        their valid times."""
        latents = [c for _, c in self.context]
        return {z: torch.cat([c[z] for c in latents], dim=2) for z in latents[0]}, [s for s, _ in self.context]

    def train_step(self, x: torch.Tensor, seconds: float) -> Dict:
        """Train on the normalized patch ``x`` (n_fine, nlev) valid at
        ``seconds`` once the context is full, then add its frozen latent to the
        context. Returns the per-level losses for the log, empty while the
        context fills."""
        stats = {}
        self.model.train()
        with torch.device(self.device):
            levels = self.model.to_levels(x)
            with torch.no_grad():
                latent = self.model.encode_frozen(levels, seconds)
            if len(self.context) == self.context.maxlen:
                stats = self._update(levels, latent, seconds)
            self.context.append((seconds, latent))
        return stats

    def _update(self, levels: Levels, latent: Levels, seconds: float) -> Dict:
        # The live window ends at c̄_t; the batch adds windows drawn from the replay buffer.
        context, context_seconds = self._stacked_context()
        live = Window(context, latent, context_seconds + [seconds])
        batch = stack_windows([live] + self.replay_buffer.sample())

        recon, forecast = self.ddp_model(levels, seconds, batch.context, batch.seconds)
        loss_recon, recon_parts = self._loss(recon, levels)
        loss_latent, latent_parts = self._loss(forecast, batch.target)

        self.optimizer.zero_grad(set_to_none=True)
        (loss_recon + loss_latent).backward()
        # Gradients are averaged over ranks, so all ranks take the same branch.
        norms = [nn.utils.clip_grad_norm_(group["params"], self.grad_clip) for group in self.optimizer.param_groups]
        if all(torch.isfinite(norm) for norm in norms):
            self.optimizer.step()
            self.model.frozen.update_parameters(self.model.reconstructor)
        self.replay_buffer.add(live)

        # Each loss per level, with the loss of outputting zero (x_t) or of persistence (c̄_{t-1} for c̄_t).
        persistence = {z: batch.target[z] - batch.context[z][:, :, -1:] for z in batch.target}
        return {
            "recon": recon_parts,
            "state_ms": self._mean_square(levels),
            "latent_mse": latent_parts,
            "latent_persist": self._mean_square(persistence),
        }

    @torch.no_grad()
    def predict_increment(self, seconds: float):
        """Change of the field from the latest context state to ``seconds``,
        D̄(ĉ) - D̄(c̄_latest), both through the frozen decoder: (n_fine, nlev)
        normalized, or ``None`` while the context fills."""
        if len(self.context) < self.context.maxlen:
            return None
        self.model.eval()
        context, context_seconds = self._stacked_context()
        last_seconds, last_latent = self.context[-1]
        with torch.device(self.device):
            forecast = self.model.decode_frozen(self.model.forecast(context, [context_seconds + [seconds]]), seconds)
            last = self.model.decode_frozen(last_latent, last_seconds)
            return self.model.to_field({z: forecast[z] - last[z] for z in forecast})
