"""Online FieldSpace autoencoder forecaster on one rank's nested ICON patch:
one model, one loss, one optimizer.

Levels
------
``levels`` are R2B levels, backbone first, e.g. ``[4, 6, 7, 8]`` on an R2B8
run; zoom ``z`` is ``level - levels[0]``. A field is split into the backbone
mean and one zero-mean residual per finer level with FieldSpaceNN's `to_zoom`
and `encode_zooms`, and summed back with `decode_zooms`. The levels are
orthogonal, so the sum of the per-level MSEs (FieldSpaceNN's `MGMultiLoss`)
is the MSE of the whole field.

Model
-----
    x_t -> encoder E -> c_t -> decoder D -> x̂_t       (`MG_AutoEncoder`, staged
                                                        compression down to
                                                        ``latent_level``)
    c_{t-H} .. c_{t-1} -> processor P -> ĉ_t          (FieldSpaceNN attention
                                                        blocks on the latent levels)

Training, at every sample step once the history is full
-------------------------------------------------------
    L = L_recon + L_incre
    L_recon = MSE(D(E(x_t)), x_t)
    L_incre = MSE(D(ĉ_t) - D(c_{t-1}), x_t - x_{t-1})

The increment is what the plugin writes to ICON, and the forecast is decoded
by the same, live decoder, so L_incre trains the processor *and* the decoder:
the decoder learns to turn latent changes into realistic field changes. The
cached latents are detached, so the encoder is trained by L_recon alone.

Both terms are FieldSpaceNN's plain `MGMultiLoss` (MSE weight 1 per level, as
in its ``mg_autoencoder.yaml`` and the Field-Space Autoencoder paper) on the
already standardized fields: the whole-field MSE, so each level counts by its
share of the variance and the increment term weighs what the 10-minute change
weighs, a few percent of the state. Reconstruction therefore leads and the
forecast follows. (Each term divided by its own target's mean square gave the
increment tens of times more weight; it starved the reconstruction, and the
latent space reorganized under the processor mid-run, job 28080773.)

Logged per step, unscaled: ``loss`` (the trained total) and the per-level
parts of both terms (``recon``, ``incre``), ``oracle`` (``incre`` had the
processor predicted ĉ_t = E(x_t) exactly: the best the current autoencoder
allows), and two references, ``persist_mse`` (per-level mean square of
x_t - x_{t-1}, the MSE of persistence) and ``state_ms`` (of x_t, the MSE of
outputting nothing). Ratios such as incre / persist_mse = 1 - skill are left
to the monitor. The backbone is never compressed.
"""

from collections import deque
from typing import Dict, List, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from fieldspacenn.src.models.mg_autoencoder.mg_autoencoder import MG_AutoEncoder
from fieldspacenn.src.models.mg_transformer.mg_base_model import create_encoder_decoder_block
from fieldspacenn.src.modules.field_space.field_space_attention import FieldSpaceAttentionConfig
from fieldspacenn.src.modules.field_space.field_space_layer import FieldSpaceLayerConfig
from fieldspacenn.src.modules.grids.grid_utils import decode_zooms, encode_zooms, to_zoom
from fieldspacenn.src.utils.losses import MGMultiLoss

# {zoom: (b=1, v=1, t, n_zoom, d=nlev, f)}, FieldSpaceNN's layout.
Levels = Dict[int, torch.Tensor]

# TimeEmbedder periods in days, as in FieldSpaceNN's configs/embedding/default.yaml.
_TIME_SCALES_DAYS = [1.0, 7.0, 30.4375, 365.25]


def _sample_configs(n_past: int, n_future: int, zooms: Sequence[int]) -> Dict[int, Dict]:
    """Whole patch as one sample, ``n_past`` + ``n_future`` timesteps."""
    conf = {"n_past_ts": n_past, "n_future_ts": n_future, "zoom_patch_sample": -1, "mask_n_last_ts": n_future}
    return {zoom: dict(conf) for zoom in zooms}


def _time_embedding(embed_dim: int) -> Dict:
    return {
        "embed_names": ["TimeEmbedder"],
        "embed_mode": "sum",
        "embed_confs": {
            "TimeEmbedder": {
                "in_channels": 1,
                "embed_dim": embed_dim,
                "time_scales": _TIME_SCALES_DAYS,
                "time_min": 0.0,
                "time_max": 1.0,
                "use_linear": False,
                # FieldSpaceNN's shared-embedder cache is a process-wide singleton.
                "shared": False,
            }
        },
    }


class FieldSpaceAE(nn.Module):
    """Encoder, decoder and latent processor of one rank's patch."""

    def __init__(
        self,
        mgrids,
        levels: Sequence[int],
        latent_level: int,
        nlev: int,
        n_history: int = 4,
        latent_channels: int = 4,
        n_blocks: int = 2,
        n_processor_blocks: int = 2,
        att_dim: int = 128,
        n_head_channels: int = 8,
        hidden_dim: int = 64,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.zooms = [level - levels[0] for level in levels]
        self.fine_zoom = self.zooms[-1]
        self.nlev = nlev
        self.all_zooms = list(range(len(mgrids)))
        self.step = _sample_configs(1, 0, self.all_zooms)
        self.window = _sample_configs(n_history, 1, self.all_zooms)
        time = _time_embedding(time_embed_dim)

        # All blocks are "ext" (the legacy blocks reject levels with different
        # channel counts), with attention tokens on the backbone.
        def attention(zooms, token_len_time=1):
            return FieldSpaceAttentionConfig(
                token_zoom=0, q_zooms=list(zooms), kv_zooms=list(zooms), att_dim=att_dim,
                n_head_channels=n_head_channels, token_len_time=token_len_time,
                embed_confs=time, block_type="ext",
            )

        def field_layer(in_zooms, target_zooms, out_zooms, target_features):
            return FieldSpaceLayerConfig(
                in_zooms=in_zooms, target_zooms=target_zooms, field_zoom=in_zooms[0], out_zooms=out_zooms,
                target_features=target_features, type="mlp", block_type="ext", hidden_dim=hidden_dim,
            )

        # Staged compression: the finest residual level is folded into the next
        # coarser one, stage by stage, down to the latent level, with n_blocks
        # attention blocks before and after each stage.
        latent_zoom = latent_level - levels[0]
        stages = [(fine, parent) for parent, fine in zip(self.zooms[1:], self.zooms[2:]) if fine > latent_zoom][::-1]
        zooms = list(self.zooms)
        encoder = {f"att_in_{i}": attention(zooms) for i in range(n_blocks)}
        for s, (fine, parent) in enumerate(stages):
            zooms.remove(fine)
            encoder[f"compress_{s}"] = field_layer([parent, fine], [parent], list(zooms), latent_channels)
            encoder.update({f"att_{s}_{i}": attention(zooms) for i in range(n_blocks)})
        self.latent_features = {z: latent_channels if z in dict(stages).values() else 1 for z in zooms}
        decoder = {f"latent_att_{i}": attention(zooms) for i in range(n_blocks)}
        for s, (fine, parent) in reversed(list(enumerate(stages))):
            zooms = sorted(zooms + [fine])
            fine_channels = latent_channels if s > 0 else 1  # fine was itself a parent in the stage before
            decoder[f"decompress_{s}"] = field_layer([parent], [parent, fine], list(zooms), {parent: 1, fine: fine_channels})
            decoder.update({f"att_{s}_{i}": attention(zooms) for i in range(n_blocks)})
        self.autoencoder = MG_AutoEncoder(
            mgrids=mgrids, in_zooms=self.zooms, encoder_block_configs=encoder,
            decoder_block_configs=decoder, n_head_channels=n_head_channels,
        )

        # Processor: n_history latents plus one future slot (a copy of the
        # latest, FieldSpaceNN's "repeat" convention) form one token per
        # backbone cell. The gates start near zero, so the untrained forecast
        # is persistence.
        latent_zooms = sorted(self.latent_features)
        self.processor = nn.ModuleList(
            create_encoder_decoder_block(
                attention(latent_zooms, token_len_time=n_history + 1),
                latent_zooms,
                [self.latent_features[z] for z in latent_zooms],
                [1],
                grid_layers=self.autoencoder.grid_layers,
                n_head_channels=n_head_channels,
            )
            for _ in range(n_processor_blocks)
        )

    def emb(self, unix_seconds: Sequence[float]):
        """Time embedding input for valid times along the time axis, on the
        default device (the trainer sets it). A new dict every call:
        FieldSpaceNN blocks add keys to it."""
        days = torch.tensor([[s / 86400.0 for s in unix_seconds]])
        return {"TimeEmbedder": {zoom: days for zoom in self.all_zooms}}

    def to_levels(self, x: torch.Tensor) -> Levels:
        """(n_fine, nlev) -> backbone mean and zero-mean residuals per level."""
        x = x.reshape(1, 1, 1, -1, self.nlev, 1).clone()  # encode_zooms subtracts in place
        return encode_zooms({z: to_zoom(x, self.fine_zoom, z)[0] for z in self.zooms}, self.step, None)

    def to_field(self, levels: Levels) -> torch.Tensor:
        """Inverse of :meth:`to_levels`."""
        return decode_zooms(dict(levels), self.step, self.fine_zoom)[self.fine_zoom].reshape(-1, self.nlev)

    # FieldSpaceNN blocks write their outputs into the input dict: pass copies.
    def encode(self, levels: Levels, seconds: float) -> Levels:
        return self.autoencoder.ae_encode([dict(levels)], sample_configs=self.step, emb_groups=[self.emb([seconds])])[0]

    def decode(self, latent: Levels, seconds: float) -> Levels:
        return self.autoencoder.ae_decode([dict(latent)], sample_configs=self.step, emb_groups=[self.emb([seconds])])[0]

    def forecast(self, history: Levels, history_seconds: List[float], seconds: float) -> Levels:
        x = [{z: torch.cat([h, h[:, :, -1:]], dim=2) for z, h in history.items()}]
        emb = self.emb(history_seconds + [seconds])
        for block in self.processor:
            x = block(x, sample_configs=self.window, emb_groups=[emb])
        return {z: t[:, :, -1:] for z, t in x[0].items()}

    def forward(self, levels: Levels, seconds: float, history: Levels, history_seconds: List[float], prev_seconds: float):
        """Training pass, one call so DDP sees every parameter: the latent and
        reconstruction of x_t, the decoded forecast of x_t and the decode of
        the newest history latent c_{t-1}."""
        latent = self.encode(levels, seconds)
        recon = self.decode(latent, seconds)
        forecast = self.decode(self.forecast(history, history_seconds, seconds), seconds)
        prev = self.decode({z: h[:, :, -1:] for z, h in history.items()}, prev_seconds)
        return latent, recon, forecast, prev


class OnlineFieldSpaceAE:
    """Online training of :class:`FieldSpaceAE` on one rank's patch, one call
    to :meth:`train_step` per sample step. See the module docstring."""

    def __init__(
        self,
        mgrids,
        levels: Sequence[int],
        latent_level: int,
        nlev: int,
        n_history: int = 4,
        lr: float = 2e-4,
        grad_clip: float = 1.0,
        device: torch.device = torch.device("cuda", 0),
        **model_kwargs,
    ):
        self.device, self.grad_clip = device, grad_clip
        # FieldSpaceNN creates device-less tensors in GridLayer and in forward
        # passes; the grids must be on the same device as those.
        mgrids = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in grid.items()} for grid in mgrids]
        with torch.device(device):
            self.model = FieldSpaceAE(mgrids, levels, latent_level, nlev, n_history, **model_kwargs)
        self.ddp_model = (
            DDP(self.model, broadcast_buffers=False)  # grid buffers differ: patch sizes differ
            if dist.is_initialized() and dist.get_world_size() > 1
            else self.model
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.level_loss = MGMultiLoss({str(z): {"MSE_loss": 1.0} for z in self.model.zooms})
        self.history = deque(maxlen=n_history)  # (seconds, levels, detached latent) per sample step

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
        """Cached latents stacked on the time axis, oldest first, and their valid times."""
        latents = [c for _, _, c in self.history]
        return {z: torch.cat([c[z] for c in latents], dim=2) for z in latents[0]}, [s for s, _, _ in self.history]

    def train_step(self, x: torch.Tensor, seconds: float) -> Dict:
        """Train on the normalized patch ``x`` (n_fine, nlev) valid at
        ``seconds`` once the history is full, then cache its latent. Returns
        the per-level losses, empty while the history fills."""
        stats = {}
        self.model.train()
        with torch.device(self.device):
            levels = self.model.to_levels(x)
            if len(self.history) == self.history.maxlen:
                stats, latent = self._update(levels, seconds)
            else:
                with torch.no_grad():
                    latent = self.model.encode(levels, seconds)
            self.history.append((seconds, levels, {z: c.detach() for z, c in latent.items()}))
        return stats

    def _update(self, levels: Levels, seconds: float):
        history, history_seconds = self._history()
        prev_seconds, prev_levels, _ = self.history[-1]
        increment = {z: levels[z] - prev_levels[z] for z in levels}

        stats = {"persist_mse": self._mean_square(increment), "state_ms": self._mean_square(levels)}
        latent, recon, forecast, prev = self.ddp_model(levels, seconds, history, history_seconds, prev_seconds)
        loss_recon, stats["recon"] = self._loss(recon, levels)
        loss_incre, stats["incre"] = self._loss({z: forecast[z] - prev[z] for z in levels}, increment)
        with torch.no_grad():
            _, stats["oracle"] = self._loss({z: recon[z] - prev[z] for z in levels}, increment)
        loss = loss_recon + loss_incre
        stats["loss"] = float(loss)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradients are averaged over ranks, so all ranks take the same branch.
        norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        if torch.isfinite(norm):
            self.optimizer.step()
        stats["grad_norm"] = float(norm)
        return stats, latent

    @torch.no_grad()
    def predict_increment(self, seconds: float):
        """Change of the field from the latest cached state to ``seconds``,
        D(ĉ) - D(c_latest): (n_fine, nlev) normalized, or ``None`` while the
        history fills."""
        if len(self.history) < self.history.maxlen:
            return None
        self.model.eval()
        history, history_seconds = self._history()
        last_seconds, _, last_latent = self.history[-1]
        with torch.device(self.device):
            forecast = self.model.decode(self.model.forecast(history, history_seconds, seconds), seconds)
            last = self.model.decode(last_latent, last_seconds)
            return self.model.to_field({z: forecast[z] - last[z] for z in forecast})
