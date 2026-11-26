"""Model-independent training bookkeeping shared by the plugins: normalization
statistics from the dry run, checkpoints, and averaging of log rows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

# ----------------------------------------------------------------------------
# Online normalization statistics
# ----------------------------------------------------------------------------


@dataclass
class NormalizationStats:
    mean: torch.Tensor
    std: torch.Tensor


class RunningMeanStd:
    """Accumulates per-node mean/std over a fixed number of samples, then freezes.

    Pure tensor bookkeeping — the caller is responsible for feeding it one
    already-extracted sample per step via :meth:`update`.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        n_samples: int,
        device: torch.device,
        eps: float = 1e-6,
    ) -> None:
        self.n_samples = int(n_samples)
        self.eps = eps
        self.count = 0
        self._sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self._sumsq = torch.zeros(shape, dtype=torch.float64, device=device)
        self.stats: Optional[NormalizationStats] = None

    @property
    def done(self) -> bool:
        return self.stats is not None

    def update(self, x: torch.Tensor) -> bool:
        """Accumulate one sample. Returns True once statistics are frozen.

        No-op (returns True immediately) once already done, so callers can
        call this unconditionally every step without checking ``done`` first.
        """
        if self.done:
            return True
        x64 = x.double()
        self._sum += x64
        self._sumsq += x64 * x64
        self.count += 1
        if self.count >= self.n_samples:
            mean = (self._sum / self.count).float()
            e_x2 = (self._sumsq / self.count).float()
            var = torch.clamp(e_x2 - mean * mean, min=self.eps)
            self.stats = NormalizationStats(mean=mean, std=torch.sqrt(var))
        return self.done

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        assert self.stats is not None, "RunningMeanStd: statistics not finalized yet"
        return (x - self.stats.mean) / self.stats.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        assert self.stats is not None, "RunningMeanStd: statistics not finalized yet"
        return x * self.stats.std + self.stats.mean



def mean_stats(rows: Sequence[Dict]) -> Dict:
    """Mean over log rows with the same keys, whose values are numbers or
    ``{level: number}`` dicts (e.g. a trainer's per-step stats)."""
    means = {}
    for key, value in rows[0].items():
        if isinstance(value, dict):
            means[key] = {level: sum(row[key][level] for row in rows) / len(rows) for level in value}
        else:
            means[key] = sum(row[key] for row in rows) / len(rows)
    return means


# ----------------------------------------------------------------------------
# Checkpoint load/save helpers
# ----------------------------------------------------------------------------


def save_checkpoint(
    trainer: Any,
    checkpoint_path: str,
    compute_rank: int,
    step: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model + optimizer state to ``checkpoint_path``.

    ``trainer`` is any object exposing ``.model`` and ``.optimizer``
    (``gnn.model.OnlineGNNTrainer``, ``unet.model.OnlineUNetTrainer``, ...).
    Only compute rank 0 writes to avoid concurrent writes on the shared
    filesystem; all other GPU ranks return immediately.

    ``extra`` is merged into the saved dict, for state a checkpoint is useless
    without but that does not live on the model - notably the normalization
    mean/std, which are derived from the dry run and cannot be recovered from
    the weights. Keys never override the three written here.
    """
    if compute_rank != 0:
        return
    payload = dict(extra or {})
    payload.update(
        {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "step": step,
        }
    )
    # Write to a temporary file first, then rename for an atomic replace.
    tmp_path = checkpoint_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def rollback_checkpoint(trainer: Any, checkpoint_path: str) -> int:
    """Reload model + optimizer from the last saved checkpoint on all GPU ranks.

    Returns the step stored in the checkpoint, or 0 if no checkpoint exists.
    All compute ranks reload the same file so weights and optimizer moments stay
    consistent after a NaN-triggered recovery.
    """
    if not os.path.exists(checkpoint_path):
        return 0
    ckpt = torch.load(
        checkpoint_path,
        map_location=torch.device("cuda", 0),
        weights_only=False,
    )
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt.get("step", 0))
