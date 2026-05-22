from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Standard 3-level UNet with skip connections.

    Input/output shape: (B, channels, H, W).
    Requires H and W to be divisible by 8 (three MaxPool halvings).
    At HEALPix level 6: nside=64, which is exactly divisible.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        b = base_channels

        self.enc1 = DoubleConv(in_channels, b)
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(b, b * 2)
        self.down2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.down3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(b * 4, b * 8)

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)

        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))

        h = self.bottleneck(self.down3(s3))

        h = self.dec3(torch.cat([self.up3(h), s3], dim=1))
        h = self.dec2(torch.cat([self.up2(h), s2], dim=1))
        h = self.dec1(torch.cat([self.up1(h), s1], dim=1))

        return self.out_conv(h)


@dataclass
class UNetSnapshot:
    faces: torch.Tensor  # (faces_per_rank, nlev, nside, nside) — float32, on GPU
    unix_seconds: float


class OnlineUNetTrainer:
    def __init__(
        self,
        nlev: int,
        lr: float = 2e-4,
        base_channels: int = 64,
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

        self.model = UNet(
            in_channels=self.nlev,
            out_channels=self.nlev,
            base_channels=base_channels,
        ).to(self.device)

        self.forward_model = self._wrap_ddp(use_ddp)

        self.optimizer = torch.optim.Adam(
            self.forward_model.parameters(), lr=lr, weight_decay=0.0
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        self.log_fn(
            f"[rank={rank}] UNet initialized: nlev={nlev}, base_channels={base_channels}, "
            f"params={n_params:,}, device={self.device}, ddp={isinstance(self.forward_model, DDP)}"
        )

    def _wrap_ddp(self, use_ddp: Optional[bool]) -> torch.nn.Module:
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

        pred = self.forward_model(source.faces)
        loss = F.mse_loss(pred, target.faces)
        loss.backward()

        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        return {"loss": loss.item(), "loss_dict": {"train/MSE": loss.item()}}
