from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def _ensure_fieldspacenn_on_path() -> None:
    env_root = os.environ.get("FIELDSPACENN_ROOT", "")
    if env_root:
        fieldspacenn_root = Path(env_root).resolve()
    else:
        src_root = Path(__file__).resolve().parents[2]
        fieldspacenn_root = src_root / "FieldSpaceNN"
    if fieldspacenn_root.exists():
        path = str(fieldspacenn_root)
        if path not in sys.path:
            sys.path.insert(0, path)


def _build_fscnn(nlev: int, model_channels: int = 64) -> nn.Module:
    from fieldspacenn.src.models.cnn.model import CNN
    from fieldspacenn.src.models.cnn.confs import CNNBlockConfig, PatchEmbConfig

    block_configs = [
        CNNBlockConfig(
            depth=1, block_type="ResnetBlock", ch_mult=2, enc=True,
            sub_confs={"blocks": ["normal", "normal", "down"]},
        ),
        CNNBlockConfig(
            depth=3, block_type="ResnetBlock", ch_mult=2, enc=True,
            sub_confs={"blocks": ["normal", "normal", "down"]},
        ),
        CNNBlockConfig(
            depth=3, block_type="ResnetBlock", ch_mult=2,
            sub_confs={"blocks": ["normal"]},
        ),
        CNNBlockConfig(
            depth=4, block_type="ResnetBlock", ch_mult=0.5, dec=True,
            sub_confs={"blocks": ["normal", "normal", "up"]},
        ),
    ]
    patch_emb_config = PatchEmbConfig(
        block_type="ConvBlock",
        patch_emb_size=(1, 1),
        patch_emb_kernel=(3, 3),
    )
    return CNN(
        init_in_ch=nlev,
        final_out_ch=nlev,
        block_configs=block_configs,
        patch_emb_config=patch_emb_config,
        model_channels=model_channels,
        skip_connections=True,
        spatial_dim_count=2,
    )


@dataclass
class UNetSnapshot:
    faces: torch.Tensor  # (faces_per_rank, nlev, nside, nside) — float32, on GPU
    unix_seconds: float


class OnlineUNetTrainer:
    def __init__(
        self,
        nlev: int,
        lr: float = 2e-4,
        model_channels: int = 64,
        grad_clip: Optional[float] = 1.0,
        use_ddp: Optional[bool] = None,
        device: Optional[torch.device] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        rank: Optional[int] = None,
    ) -> None:
        _ensure_fieldspacenn_on_path()

        self.nlev = int(nlev)
        self.device = torch.device("cuda", 0) if device is None else device
        self.grad_clip = grad_clip
        self.rank = rank
        self.log_fn = log_fn if log_fn is not None else (lambda msg: None)

        self.model = _build_fscnn(nlev, model_channels).to(self.device)
        self.forward_model = self._wrap_ddp(use_ddp)
        self.optimizer = torch.optim.Adam(
            self.forward_model.parameters(), lr=lr, weight_decay=0.0
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        self.log_fn(
            f"[rank={rank}] FieldSpaceNN CNN initialized: nlev={nlev}, "
            f"model_channels={model_channels}, params={n_params:,}, "
            f"device={self.device}, ddp={isinstance(self.forward_model, DDP)}"
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

    @staticmethod
    def _to_cnn_input(faces: torch.Tensor) -> torch.Tensor:
        # (F, nlev, H, W) → (F, 1, 1, H, W, nlev)
        return faces.permute(0, 2, 3, 1)[:, None, None]

    @staticmethod
    def _from_cnn_output(out: torch.Tensor) -> torch.Tensor:
        # (F, 1, 1, H, W, nlev) → (F, nlev, H, W)
        return out[:, 0, 0].permute(0, 3, 1, 2)

    @torch.no_grad()
    def prepare_snapshot(
        self,
        ua_faces: torch.Tensor,
        unix_seconds: float,
    ) -> UNetSnapshot:
        faces = ua_faces.to(self.device, dtype=torch.float32, non_blocking=True).detach()
        return UNetSnapshot(faces=faces, unix_seconds=float(unix_seconds))

    def train_step(
        self,
        source: UNetSnapshot,
        target: UNetSnapshot,
    ) -> Dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        pred = self.forward_model(self._to_cnn_input(source.faces), emb=None)
        pred = self._from_cnn_output(pred)
        loss = F.mse_loss(pred, target.faces)
        loss.backward()

        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        return {"loss": loss.item(), "loss_dict": {"train/MSE": loss.item()}}
