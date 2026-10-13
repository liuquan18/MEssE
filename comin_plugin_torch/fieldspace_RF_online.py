"""Online RECONSTRUCTOR + FORECASTOR on one rank's nested ICON patch
(`.claude/architecture_loss.pdf`).

Levels
------
``levels`` are R2B levels, backbone first, e.g. ``[4, 6, 7, 8]`` on an R2B8
run; zoom ``z`` is ``level - levels[0]``. A field is split into the backbone
mean and one zero-mean residual per finer level with FieldSpaceNN's `to_zoom`
and `encode_zooms`, and summed back with `decode_zooms`. Both losses below are
FieldSpaceNN's `MGMultiLoss` on these levels, one MSE per level with weight 1
(as in FieldSpaceNN's configs): the whole-field MSE, so each level counts by
its share of the variance.

RECONSTRUCTOR
    x_t -> encoder -> c_t -> decoder -> x̂_t, an `MG_AutoEncoder` with the
    staged compression of `.claude/compression.png`: each residual level finer
    than ``latent_level`` is folded into the next coarser one. Trained with
    ``L_recon`` = per-level MSE(x̂_t, x_t).

FORECASTOR
    c_{t-H} .. c_{t-1} -> MG transformer -> ĉ_t -> frozen decoder. Trained with
    ``L_incre`` = per-level MSE(D(ĉ_t) - D(c_{t-1}), x_t - x_{t-1}): the
    increment from the most recent state, which is also what is written to
    ICON. Averaged over a batch of windows (latent replay, below).

The two are trained independently: the forecaster's loss reaches only the
forecaster. The frozen encoder/decoder is an exponential moving average of the
reconstructor (`torch.optim.swa_utils.AveragedModel`). It encodes the cached
history c_{t-H} .. c_{t-1} and decodes the forecast, so the forecaster sees a
latent space that moves slowly and consistently while the reconstructor keeps
learning.

Latent replay
-------------
Against forgetting earlier weather while training on a correlated stream. A
training window is the frozen latents c_{t-H} .. c_{t-1}, the true increment
x_t - x_{t-1} and their valid times. The last ``replay_size`` windows are
kept; every update draws ``replay_batch`` of them at random, and the
forecaster is trained on these plus the new window as one batch. The stored
latents are not re-encoded: the buffer length bounds how much older the
encoder that made them is.

At sample step t (history full)
-------------------------------
1. ``L_recon`` on x_t plus ``L_incre`` on the batch, one backward, one
   optimizer step (separate learning rates and gradient clipping per model),
   then the EMA update.
2. The new window is stored for replay, c_t = E_frozen(x_t) is cached, and the
   plugin asks for the increment to t+1.
3. Returned for the log, per level: both losses and their references,
   ``state_ms`` (mean square of x_t, the loss of outputting zero) and
   ``persist_mse`` (mean square of the batch's increments, the loss of
   persistence).
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

    history: Levels  # frozen latents c_{t-H} .. c_{t-1}
    increment: Levels  # x_t - x_{t-1}, the target
    seconds: List[float]  # valid times of the history, then of x_t


def stack(windows: Sequence[Window]) -> Window:
    """Windows concatenated on the batch axis; ``seconds`` becomes one row per
    window."""
    return Window(
        {z: torch.cat([w.history[z] for w in windows]) for z in windows[0].history},
        {z: torch.cat([w.increment[z] for w in windows]) for z in windows[0].increment},
        [w.seconds for w in windows],
    )


class LatentReplay:
    """The last ``size`` training windows; :meth:`sample` draws up to
    ``batch`` of them at random, without replacement. ``size=0`` turns replay
    off."""

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


class _Blocks:
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


def autoencoder_configs(zooms: List[int], latent_zoom: int, channels: int, n_blocks: int, blocks: _Blocks):
    """Encoder and decoder block configs of the staged compression, and the
    channels of every latent level. The finest residual level is folded into
    the next coarser one, stage by stage, down to ``latent_zoom`` (``channels``
    per latent cell), with ``n_blocks`` attention blocks before and after each
    stage."""
    stages = [(fine, parent) for parent, fine in zip(zooms[1:], zooms[2:]) if fine > latent_zoom][::-1]
    zooms = list(zooms)
    encoder = {f"att_in_{i}": blocks.attention(zooms) for i in range(n_blocks)}
    for s, (fine, parent) in enumerate(stages):
        zooms.remove(fine)
        encoder[f"compress_{s}"] = blocks.field_layer([parent, fine], [parent], parent, list(zooms), channels)
        encoder.update({f"att_{s}_{i}": blocks.attention(zooms) for i in range(n_blocks)})
    latent_features = {z: channels if z in dict(stages).values() else 1 for z in zooms}
    decoder = {f"latent_att_{i}": blocks.attention(zooms) for i in range(n_blocks)}
    for s, (fine, parent) in reversed(list(enumerate(stages))):
        zooms = sorted(zooms + [fine])
        fine_channels = channels if s > 0 else 1  # fine was itself a parent in the stage before
        decoder[f"decompress_{s}"] = blocks.field_layer(
            [parent], [parent, fine], parent, list(zooms), {parent: 1, fine: fine_channels}
        )
        decoder.update({f"att_{s}_{i}": blocks.attention(zooms) for i in range(n_blocks)})
    return encoder, decoder, latent_features


class Reconstructor(nn.Module):
    """Encoder/decoder between a field's levels and its latent."""

    def __init__(self, mgrids, zooms, latent_zoom, channels, n_blocks, blocks: _Blocks):
        super().__init__()
        encoder, decoder, self.latent_features = autoencoder_configs(zooms, latent_zoom, channels, n_blocks, blocks)
        self.ae = MG_AutoEncoder(
            mgrids=mgrids,
            in_zooms=zooms,
            encoder_block_configs=encoder,
            decoder_block_configs=decoder,
            n_head_channels=blocks.n_head_channels,
        )
        self.step = _sample_configs(1, 0, range(len(mgrids)))

    # FieldSpaceNN blocks write their outputs into the input dict: pass copies.
    def encode(self, levels: Levels, emb) -> Levels:
        return self.ae.ae_encode([dict(levels)], sample_configs=self.step, emb_groups=[emb])[0]

    def decode(self, latent: Levels, emb) -> Levels:
        return self.ae.ae_decode([dict(latent)], sample_configs=self.step, emb_groups=[emb])[0]


class Forecaster(nn.Module):
    """MG transformer over the latent levels: ``n_history`` latents plus one
    future slot (a copy of the latest, FieldSpaceNN's ``repeat`` convention)
    form one token per backbone cell; the future slot is returned. The gates
    start near zero, so the untrained forecast is persistence."""

    def __init__(self, grid_layers, latent_features: Dict[int, int], n_history, n_blocks, blocks: _Blocks):
        super().__init__()
        zooms = sorted(latent_features)
        self.blocks = nn.ModuleList(
            create_encoder_decoder_block(
                blocks.attention(zooms, token_len_time=n_history + 1),
                zooms,
                [latent_features[z] for z in zooms],
                [1],
                grid_layers=grid_layers,
                n_head_channels=blocks.n_head_channels,
            )
            for _ in range(n_blocks)
        )
        self.window = _sample_configs(n_history, 1, range(len(grid_layers)))

    def forward(self, history: Levels, emb) -> Levels:
        x = [{z: torch.cat([h, h[:, :, -1:]], dim=2) for z, h in history.items()}]
        for block in self.blocks:
            x = block(x, sample_configs=self.window, emb_groups=[emb])
        return {z: t[:, :, -1:] for z, t in x[0].items()}


class ReconstructorForecaster(nn.Module):
    """Both models plus the frozen (EMA) copy of the reconstructor."""

    def __init__(
        self,
        mgrids,
        levels: Sequence[int],
        latent_level: int,
        nlev: int,
        n_history: int = 4,
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
        self.step = _sample_configs(1, 0, self.all_zooms)
        blocks = _Blocks(att_dim, n_head_channels, hidden_dim, time_embed_dim)

        self.reconstructor = Reconstructor(
            mgrids, self.zooms, latent_level - levels[0], latent_channels, n_blocks, blocks
        )
        self.forecaster = Forecaster(
            self.reconstructor.ae.grid_layers, self.reconstructor.latent_features, n_history, n_forecaster_blocks, blocks
        )
        self.frozen = AveragedModel(self.reconstructor, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        self.frozen.requires_grad_(False)

    def emb(self, times: Sequence[Sequence[float]]):
        """Time embedding input: one row of valid times (unix seconds) along
        the time axis per batch sample, on the default device (the trainer
        sets it). A new dict every call: FieldSpaceNN blocks add keys to it."""
        days = torch.tensor([[s / 86400.0 for s in row] for row in times])
        return {"TimeEmbedder": {zoom: days for zoom in self.all_zooms}}

    def to_levels(self, x: torch.Tensor) -> Levels:
        """(n_fine, nlev) -> backbone mean and zero-mean residuals per level."""
        x = x.reshape(1, 1, 1, -1, self.nlev, 1).clone()  # encode_zooms subtracts in place
        return encode_zooms({z: to_zoom(x, self.fine_zoom, z)[0] for z in self.zooms}, self.step, None)

    def to_field(self, levels: Levels) -> torch.Tensor:
        """Inverse of :meth:`to_levels`."""
        return decode_zooms(dict(levels), self.step, self.fine_zoom)[self.fine_zoom].reshape(-1, self.nlev)

    def encode_frozen(self, levels: Levels, seconds: float) -> Levels:
        return self.frozen.module.encode(levels, self.emb([[seconds]]))

    def decode_frozen(self, latent: Levels, seconds: Sequence[float]) -> Levels:
        """Decode a batch of latents, one valid time per sample."""
        return self.frozen.module.decode(latent, self.emb([[s] for s in seconds]))

    def forecast(self, history: Levels, times: Sequence[Sequence[float]]) -> Levels:
        """Latent forecast of a batch of histories; ``times`` has one row per
        sample: the history's valid times, then the forecast's."""
        return self.forecaster(history, self.emb(times))

    def forward(self, levels: Levels, seconds: float, history: Levels, times: Sequence[Sequence[float]]):
        """Training pass: the reconstruction of x_t and the frozen decode of the
        forecasts of a batch of windows. One call, so DDP sees every trainable
        parameter once."""
        recon = self.reconstructor.decode(self.reconstructor.encode(levels, self.emb([[seconds]])), self.emb([[seconds]]))
        forecast = self.decode_frozen(self.forecast(history, times), [row[-1] for row in times])
        return recon, forecast


class OnlineReconstructorForecaster:
    """Online training of :class:`ReconstructorForecaster` on one rank's patch,
    one call to :meth:`train_step` per sample step. See the module docstring."""

    def __init__(
        self,
        mgrids,
        levels: Sequence[int],
        latent_level: int,
        nlev: int,
        n_history: int = 4,
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
            self.model = ReconstructorForecaster(mgrids, levels, latent_level, nlev, n_history, **model_kwargs)
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
        self.history = deque(maxlen=n_history)  # (seconds, levels, frozen latent) per sample step
        rng = torch.Generator().manual_seed(dist.get_rank() if dist.is_initialized() else 0)
        self.replay = LatentReplay(replay_size, replay_batch, rng)

    def _loss(self, output: Levels, target: Levels):
        """Plain MGMultiLoss: the trained loss (sum of the per-level MSEs =
        whole-field MSE) and its per-level parts, which are what is logged."""
        total, parts = self.level_loss(output, target)
        return total, {z: float(parts[f"level{z}_MSE_loss"]) for z in target}

    @staticmethod
    def _mean_square(levels: Levels) -> Dict[int, float]:
        """Per-level mean square: the MSE of predicting zero for ``levels``."""
        return {z: float(t.square().mean()) for z, t in levels.items()}

    def _history(self):
        """Cached frozen latents stacked on the time axis, oldest first, and
        their valid times."""
        latents = [c for _, _, c in self.history]
        return {z: torch.cat([c[z] for c in latents], dim=2) for z in latents[0]}, [s for s, _, _ in self.history]

    def train_step(self, x: torch.Tensor, seconds: float) -> Dict:
        """Train on the normalized patch ``x`` (n_fine, nlev) valid at
        ``seconds`` once the history is full, then cache its frozen latent.
        Returns the per-level losses for the log, empty while the history
        fills."""
        stats = {}
        self.model.train()
        with torch.device(self.device):
            levels = self.model.to_levels(x)
            with torch.no_grad():
                latent = self.model.encode_frozen(levels, seconds)
            if len(self.history) == self.history.maxlen:
                stats = self._update(levels, seconds)
            self.history.append((seconds, levels, latent))
        return stats

    def _update(self, levels: Levels, seconds: float) -> Dict:
        # The new window ends at x_t; the batch adds randomly drawn stored ones.
        history, history_seconds = self._history()
        prev_levels = self.history[-1][1]
        window = Window(history, {z: levels[z] - prev_levels[z] for z in levels}, history_seconds + [seconds])
        batch = stack([window] + self.replay.sample())

        recon, forecast = self.ddp_model(levels, seconds, batch.history, batch.seconds)
        with torch.no_grad():  # D(c_{t-1}) of every window, from its newest history latent
            base = self.model.decode_frozen({z: h[:, :, -1:] for z, h in batch.history.items()}, [s[-2] for s in batch.seconds])
        loss_recon, recon_parts = self._loss(recon, levels)
        loss_incre, incre_parts = self._loss({z: forecast[z] - base[z] for z in levels}, batch.increment)

        self.optimizer.zero_grad(set_to_none=True)
        (loss_recon + loss_incre).backward()
        # Gradients are averaged over ranks, so all ranks take the same branch.
        norms = [nn.utils.clip_grad_norm_(group["params"], self.grad_clip) for group in self.optimizer.param_groups]
        if all(torch.isfinite(norm) for norm in norms):
            self.optimizer.step()
            self.model.frozen.update_parameters(self.model.reconstructor)
        self.replay.add(window)

        # Each loss per level, with the loss of outputting zero (x_t) or of persistence (the batch's increments).
        return {
            "recon": recon_parts,
            "state_ms": self._mean_square(levels),
            "incre": incre_parts,
            "persist_mse": self._mean_square(batch.increment),
        }

    @torch.no_grad()
    def predict_increment(self, seconds: float):
        """Change of the field from the latest cached state to ``seconds``,
        D(ĉ) - D(c_latest), both through the frozen decoder: (n_fine, nlev)
        normalized, or ``None`` while the history fills."""
        if len(self.history) < self.history.maxlen:
            return None
        self.model.eval()
        history, history_seconds = self._history()
        last_seconds, _, last_latent = self.history[-1]
        with torch.device(self.device):
            forecast = self.model.decode_frozen(self.model.forecast(history, [history_seconds + [seconds]]), [seconds])
            last = self.model.decode_frozen(last_latent, [last_seconds])
            return self.model.to_field({z: forecast[z] - last[z] for z in forecast})
