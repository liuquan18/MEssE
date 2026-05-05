from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FaceCNN: operates on HEALPix faces treated as 2D images.
# Each HEALPix face in NEST ordering is a contiguous nside×nside spatial grid.
# Input/output: (faces_per_rank, n_channels, nside, nside)
# ---------------------------------------------------------------------------
class FaceCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (faces_per_rank, in_channels, nside, nside)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)


@dataclass
class FaceCNNSnapshot:
    faces: torch.Tensor  # (faces_per_rank, nlev, nside, nside)
    unix_seconds: float


class OnlineFaceCNNTrainer:
    """Lightweight FaceCNN trainer with the same interface as OnlineFieldSpaceNNTrainer."""

    def __init__(
        self,
        cfg: Any,
        nlev: int,
        device: Optional[torch.device] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        rank: Optional[int] = None,
    ) -> None:
        from hydra.utils import instantiate

        self.nlev = int(nlev)
        self.forecast_horizon_steps = int(cfg.online.forecast_horizon_steps)
        self.device = torch.device("cuda", 0) if device is None else device
        self.rank = rank
        self.log_fn = log_fn or (lambda msg: None)

        self.model = FaceCNN(in_channels=nlev, out_channels=nlev).to(self.device)
        self.optimizer = instantiate(cfg.optimizer, params=self.model.parameters())
        self.log_fn(
            f"[rank={rank}] FaceCNN initialized: nlev={nlev}, "
            f"horizon={self.forecast_horizon_steps}"
        )

    @torch.no_grad()
    def prepare_snapshot(
        self, ua_faces: torch.Tensor, unix_seconds: float
    ) -> FaceCNNSnapshot:
        faces = ua_faces.to(self.device, dtype=torch.float32, non_blocking=True).detach()
        return FaceCNNSnapshot(faces=faces, unix_seconds=float(unix_seconds))

    def train_step(
        self, source: FaceCNNSnapshot, target: FaceCNNSnapshot
    ) -> Dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss = F.mse_loss(self.model(source.faces), target.faces)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "loss_dict": {"train/MSE": loss.item()}}
