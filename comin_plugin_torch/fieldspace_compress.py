"""Field-Space compression / decompression over a multi-level ICON pyramid,
plus a reservoir for caching compressed states from previous timesteps.

Implements the compression/decompression blocks of Meuer et al., "Field-space
autoencoder for scalable climate emulators" (npj Artificial Intelligence 2026,
2:50), Methods eqs 3-11, generalised to an arbitrary pyramid of ICON zoom
levels with arbitrary gaps between them.

The staged architecture
-----------------------
Following the user's design sketch (`.claude/compression.png`), compression is
a *stack* of stages rather than a single map. Each stage drops the finest
remaining level and folds its information into the next-coarser level, which
comes out wider (more channels). Between stages sit Field-space attention
blocks and the MLP that does the folding -- those live in the trainer; this
module provides the folding operator and the pyramid bookkeeping.

For the target R2B8 config, pyramid ``(3, 6, 7, 8)``::

    native R2B8                                5,242,880   1.00x
    decompose -> {x3, r6, r7, r8}              6,886,400   1.31x
    stage 1   -> {x3, r6, r7' (4 ch)}          5,575,680   1.06x
    stage 2   -> {x3, r6' (4 ch)}              1,315,840   0.25x

The same code degenerates to the two-zoom R2B4/R2B3 prototype, pyramid
``(3, 4)``: one stage, patch width ``1 + 4*1 = 5``.

Why a pyramid alone is not compression
--------------------------------------
The multi-grid decomposition is a change of basis, not a reduction. The 1.31x
above is *storage* redundancy only; in *information* the pyramid is exactly
critically sampled, because every residual level is zero-mean within its parent
group and so carries ``(g-1)/g`` of its values as free DOF::

    5,120 + (63/64)*327,680 + (3/4)*1,310,720 + (3/4)*5,242,880 = 5,242,880

which is exactly ``N_fine``. Every bit of real compression must therefore be
learned or truncated -- which is what the stages above do.

Patch width
-----------
A stage folding level ``k`` into level ``k-1`` builds, per target cell, a patch
of ``c_{k-1} + g * c_k`` channels, where ``g`` is the number of level-``k``
children per level-``k-1`` cell (paper eqs 4-5). With single-channel inputs and
``g = 4`` this is the paper's own ``1 + 4``; their three-level example
``1 + 4 + 16 = 21`` is the same thing done in one step instead of two.

Relationship to zero-filling
----------------------------
Zero-filling a fine residual -- which FieldSpaceNN supports natively, and which
the paper uses for zero-shot super-resolution ("the finer residual levels are
set to zero") -- is the untrained special case of eq 6, with the weight pinned
to the ancestor-broadcast column. :func:`ancestor_broadcast_init_` sets exactly
that, so the stack reproduces the zero-fill baseline bit-for-bit. Truncation and
learned compression are one family, which makes them directly comparable at
identical latent size: the experiment then isolates "does learning the mix
help?" from "does shrinking help?".

Vertical axis
-------------
``nlev`` is carried through untouched as an independent batch-like axis, with
every map shared across it -- matching the paper's ``C = 1`` per physical field
("scale mixing occurs only within the patch"). The vertical is therefore *not*
compressed here. Deliberate for now (this project is 2D-first), but it leaves
the ``d`` axis free for a vertical bottleneck later.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# ICON's R2Bn refinement: one level of refinement quarters each cell.
REFINEMENT_FACTOR = 4


def children_between(zoom_coarse: int, zoom_fine: int) -> int:
    """Number of ``zoom_fine`` cells inside one ``zoom_coarse`` cell.

    ``4^(z_fine - z_coarse)`` -- paper eq 3's ``n_pix``, written for a pyramid
    whose consecutive levels may be more than one refinement apart (this
    project's R2B8 config jumps 3 -> 6).
    """
    if zoom_fine < zoom_coarse:
        raise ValueError(f"zoom_fine {zoom_fine} must be >= zoom_coarse {zoom_coarse}")
    return REFINEMENT_FACTOR ** (zoom_fine - zoom_coarse)


@dataclass(frozen=True)
class PyramidSpec:
    """The ladder of ICON zoom levels a model works on, coarsest first.

    ``(3, 6, 7, 8)`` is the R2B8 target config; ``(3, 4)`` the R2B4/R2B3
    prototype. Level 0 holds the coarse *mean*; every level above it holds a
    zero-mean *residual* against its parent.
    """

    zooms: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.zooms) < 2:
            raise ValueError(f"need at least two levels, got {self.zooms}")
        if any(b <= a for a, b in zip(self.zooms, self.zooms[1:])):
            raise ValueError(f"zooms must be strictly increasing, got {self.zooms}")

    @property
    def n_levels(self) -> int:
        return len(self.zooms)

    def group_size(self, level: int) -> int:
        """Children of one level-``(level-1)`` cell at level ``level``."""
        if level < 1 or level >= self.n_levels:
            raise IndexError(f"level {level} has no parent in {self.zooms}")
        return children_between(self.zooms[level - 1], self.zooms[level])

    def cells_at(self, level: int, n_fine: int) -> int:
        """Cell count at ``level`` given the finest level's cell count."""
        return n_fine // children_between(self.zooms[level], self.zooms[-1])

    def patch_in_channels(self, level: int, c_target: int, c_drop: int) -> int:
        """Paper eq 5: patch width when folding ``level`` into ``level-1``."""
        return c_target + self.group_size(level) * c_drop


# --------------------------------------------------------------------------
# Multi-scale decomposition
# --------------------------------------------------------------------------


def _group_mean(x: torch.Tensor, group: int) -> torch.Tensor:
    """Mean over each contiguous group of ``group`` cells. ``(n, c, l) -> (n/g, c, l)``."""
    n, c, l = x.shape
    return x.reshape(n // group, group, c, l).mean(dim=1)


def _broadcast(x: torch.Tensor, group: int) -> torch.Tensor:
    """Inverse of :func:`_group_mean`'s shape change: repeat each parent over
    its ``group`` children. ``(n, c, l) -> (n*g, c, l)``."""
    n, c, l = x.shape
    return x.unsqueeze(1).expand(n, group, c, l).reshape(n * group, c, l)


def decompose_pyramid(x_fine: torch.Tensor, spec: PyramidSpec) -> List[torch.Tensor]:
    """Multi-scale decomposition of a compact fine field into the pyramid.

    ``x_fine`` is ``(n_fine, nlev)`` in :class:`icon_mgrid_utils.LocalMGrid`
    compact order, where each parent's children are contiguous at every level.
    Returns one ``(n_i, 1, nlev)`` tensor per level, coarsest first: level 0 is
    the coarse mean, levels 1.. are residuals against the broadcast parent.

    Every residual level is zero-mean within its parent group **by
    construction** -- the scale-conservation property the hierarchy rests on,
    and the reason the representation is critically sampled rather than
    redundant.
    """
    x = x_fine.unsqueeze(1) if x_fine.dim() == 2 else x_fine  # (n, 1, nlev)

    means: List[torch.Tensor] = [x]
    for level in range(spec.n_levels - 1, 0, -1):
        means.insert(0, _group_mean(means[0], spec.group_size(level)))

    levels = [means[0]]
    for level in range(1, spec.n_levels):
        levels.append(means[level] - _broadcast(means[level - 1], spec.group_size(level)))
    return levels


def recompose_pyramid(levels: Sequence[torch.Tensor], spec: PyramidSpec) -> torch.Tensor:
    """Inverse of :func:`decompose_pyramid`. Returns ``(n_fine, nlev)``.

    Requires single-channel levels -- a physical field, which is what the
    decoder's final output must be.
    """
    for i, lv in enumerate(levels):
        if lv.shape[1] != 1:
            raise ValueError(f"level {i} has {lv.shape[1]} channels; recompose needs 1")

    x = levels[0]
    for level in range(1, spec.n_levels):
        x = _broadcast(x, spec.group_size(level)) + levels[level]
    return x.squeeze(1)


def enforce_conservation(
    levels: Sequence[torch.Tensor], spec: PyramidSpec
) -> List[torch.Tensor]:
    """Project every residual level back onto the zero-mean-per-parent-group
    subspace -- the "scale conservation operation" the paper applies in the
    decoder's final layer to "re-enforce the scale-conservative hierarchy
    before reconstructing the finest field".

    Implemented directly rather than via FieldSpaceNN's ``ConservativeLayer``:
    that layer operates on the full ``(b, v, t, n, d, f)`` zoom-dict layout
    inside ``MG_Transformer``, whereas what is needed here is the same
    projection on bare per-level tensors. Shoehorning them into the MG layout
    and back would be more code, not less, and would drag a ``FieldSpaceNN``
    import into an otherwise dependency-free module. The projection itself is
    unambiguous: subtract the per-group mean. The coarse mean (level 0) is left
    alone -- it is not a residual.
    """
    out = [levels[0]]
    for level in range(1, spec.n_levels):
        r = levels[level]
        group = spec.group_size(level)
        n, c, l = r.shape
        grouped = r.reshape(n // group, group, c, l)
        out.append((grouped - grouped.mean(dim=1, keepdim=True)).reshape(n, c, l))
    return out


# --------------------------------------------------------------------------
# Patch assembly (eqs 4-5) and unstacking (eqs 8, 11)
# --------------------------------------------------------------------------


def assemble_patch(target: torch.Tensor, drop: torch.Tensor, group: int) -> torch.Tensor:
    """Paper eq 4: per target cell, concatenate the target level's own channels
    with its ``group`` children's channels from the level being dropped.

    ``(n_t, c_t, l)`` + ``(n_t*g, c_d, l)`` -> ``(n_t, c_t + g*c_d, l)``. This
    is a plain reshape because ``LocalMGrid`` keeps each parent's children
    contiguous at every level.
    """
    n_t, c_t, l = target.shape
    children = drop.reshape(n_t, group * drop.shape[1], l)
    return torch.cat([target, children], dim=1)


def disassemble_patch(
    patch: torch.Tensor, c_target: int, c_drop: int, group: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`assemble_patch` -- the canonical parent-child
    unstacking of eqs 8 and 11."""
    n_t, _, l = patch.shape
    target = patch[:, :c_target, :]
    drop = patch[:, c_target:, :].reshape(n_t * group, c_drop, l)
    return target, drop


# --------------------------------------------------------------------------
# The learned maps (eqs 6, 10)
# --------------------------------------------------------------------------


class _PatchLinear(nn.Module):
    """A single shared linear map per patch, applied across cells and across
    the vertical axis. ``(n, c_in, l) -> (n, c_out, l)``."""

    def __init__(self, c_in: int, c_out: int, bias: bool = True) -> None:
        super().__init__()
        self.c_in, self.c_out = int(c_in), int(c_out)
        self.weight = nn.Parameter(torch.empty(self.c_in, self.c_out))
        self.bias = nn.Parameter(torch.zeros(self.c_out)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("ncl,co->nol", x, self.weight)
        return out if self.bias is None else out + self.bias.view(1, -1, 1)


class CompressionStage(nn.Module):
    """Paper eq 6: fold the finest level into the next-coarser one, which comes
    out with ``c_out`` channels. The finest level is removed.

    This is the "MLP" box of the design sketch.
    """

    def __init__(
        self, spec: PyramidSpec, level: int, c_target: int, c_drop: int, c_out: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.spec, self.level = spec, int(level)
        self.c_target, self.c_drop, self.c_out = int(c_target), int(c_drop), int(c_out)
        self.group = spec.group_size(level)
        self.map = _PatchLinear(
            spec.patch_in_channels(level, c_target, c_drop), c_out, bias=bias
        )

    def forward(self, levels: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        patch = assemble_patch(levels[self.level - 1], levels[self.level], self.group)
        return list(levels[: self.level - 1]) + [self.map(patch)]


class DecompressionStage(nn.Module):
    """Paper eq 10: the mirror -- expand the coarser level back into itself plus
    the finer level that was dropped."""

    def __init__(
        self, spec: PyramidSpec, level: int, c_target: int, c_drop: int, c_out: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.spec, self.level = spec, int(level)
        self.c_target, self.c_drop, self.c_out = int(c_target), int(c_drop), int(c_out)
        self.group = spec.group_size(level)
        self.map = _PatchLinear(
            c_out, spec.patch_in_channels(level, c_target, c_drop), bias=bias
        )

    def forward(self, levels: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        patch = self.map(levels[-1])
        target, drop = disassemble_patch(patch, self.c_target, self.c_drop, self.group)
        return list(levels[:-1]) + [target, drop]


@torch.no_grad()
def ancestor_broadcast_init_(
    compress: CompressionStage, decompress: DecompressionStage
) -> None:
    """Initialise a stage pair so it reproduces the **zero-fill** baseline
    exactly: the target level's channels pass through verbatim, and the dropped
    level is reconstructed as zero.

    This is the paper's zero-shot super-resolution setting expressed as a weight
    initialisation -- the concrete sense in which the learned stage *strictly
    generalises* truncation. Requires ``c_out >= c_target``. Useful both as a
    test fixture and as a warm start: training from here can only improve.
    """
    if compress.c_out < compress.c_target:
        raise ValueError(
            f"c_out {compress.c_out} < c_target {compress.c_target}: the target "
            "level cannot pass through unchanged"
        )
    for stage in (compress, decompress):
        stage.map.weight.zero_()
        if stage.map.bias is not None:
            stage.map.bias.zero_()
    for i in range(compress.c_target):
        compress.map.weight[i, i] = 1.0
        decompress.map.weight[i, i] = 1.0


class FieldSpaceAutoencoder(nn.Module):
    """Stacked compression/decompression over a pyramid -- the design sketch's
    full encoder/decoder.

    ``n_stages`` stages each drop one level, finest first. ``encode`` returns
    the remaining levels (the latent); ``decode`` rebuilds the fine field,
    applying :func:`enforce_conservation` at the end so the output always
    satisfies the scale-conservative hierarchy regardless of what the linear
    maps produced.

    This is the frozen-able unit: once trained for reconstruction, ``encode``
    becomes a fixed function, which is what makes a long-lived latent
    :class:`LatentReservoir` sound -- a latent cached at step 1,000 still means
    the same thing at step 50,000.
    """

    def __init__(
        self,
        spec: PyramidSpec,
        n_stages: int = 1,
        c_latent: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if not 1 <= n_stages <= spec.n_levels - 1:
            raise ValueError(
                f"n_stages {n_stages} out of range for a {spec.n_levels}-level pyramid"
            )
        self.spec, self.n_stages, self.c_latent = spec, int(n_stages), int(c_latent)

        compress: List[CompressionStage] = []
        decompress: List[DecompressionStage] = []
        channels = [1] * spec.n_levels  # channels per level, before any stage
        for s in range(self.n_stages):
            level = spec.n_levels - 1 - s
            c_target, c_drop = channels[level - 1], channels[level]
            compress.append(CompressionStage(spec, level, c_target, c_drop, c_latent, bias))
            decompress.insert(
                0, DecompressionStage(spec, level, c_target, c_drop, c_latent, bias)
            )
            channels[level - 1] = c_latent
        self.compress = nn.ModuleList(compress)
        self.decompress = nn.ModuleList(decompress)

    def encode(self, x_fine: torch.Tensor) -> List[torch.Tensor]:
        levels = decompose_pyramid(x_fine, self.spec)
        for stage in self.compress:
            levels = stage(levels)
        return levels

    def decode(self, levels: Sequence[torch.Tensor]) -> torch.Tensor:
        out = list(levels)
        for stage in self.decompress:
            out = stage(out)
        return recompose_pyramid(enforce_conservation(out, self.spec), self.spec)

    def forward(self, x_fine: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x_fine))

    def init_as_zero_fill(self) -> None:
        """Warm-start every stage at the zero-fill baseline."""
        for c, d in zip(self.compress, reversed(self.decompress)):
            ancestor_broadcast_init_(c, d)

    def latent_numel(self, x_fine: torch.Tensor) -> int:
        return sum(lv.numel() for lv in self.encode(x_fine))


def zero_fill_baseline(
    x_fine: torch.Tensor, spec: PyramidSpec, n_stages: int = 1
) -> torch.Tensor:
    """The untrained control arm: drop the ``n_stages`` finest residual levels
    entirely and rebuild from what is left.

    At equal latent size this is what a learned bottleneck has to beat --
    comparing the two isolates the value of *learning the mix* from the value of
    *shrinking*.
    """
    levels = decompose_pyramid(x_fine, spec)
    for level in range(spec.n_levels - n_stages, spec.n_levels):
        levels[level] = torch.zeros_like(levels[level])
    return recompose_pyramid(levels, spec)


# --------------------------------------------------------------------------
# Reservoir
# --------------------------------------------------------------------------


@dataclass
class LatentState:
    """One timestep's compressed field, as produced by
    :meth:`FieldSpaceAutoencoder.encode`."""

    levels: List[torch.Tensor]
    unix_seconds: float

    def numel(self) -> int:
        return sum(lv.numel() for lv in self.levels)


class LatentReservoir:
    """Ring buffer of compressed states from previous timesteps.

    Serves the two roles the design needs, from one store:

    * **history** -- the most recent ``k`` states, to be stacked on the ``t``
      axis as multi-timestep model input (FieldSpaceNN's ``Tokenizer`` already
      treats ``t`` as a first-class tokenizable axis, so this is what feeds it).
    * **replay** -- uniform samples from the whole buffer, to decorrelate the
      strongly autocorrelated online gradient stream.

    Everything is stored detached, so nothing keeps an autograd graph alive
    across timesteps -- the buffer holds data, never activations.

    Replay is only *sound* once the encoder is frozen. While the encoder is
    still training, a state cached at step 1,000 no longer means what a freshly
    encoded one does: the classic latent-replay staleness problem. Construct
    with ``allow_replay=False`` during the reconstruction phase, and flip it on
    when the encoder is frozen; :meth:`sample` refuses otherwise rather than
    silently returning stale latents.
    """

    def __init__(self, capacity: int, allow_replay: bool = False) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self.allow_replay = bool(allow_replay)
        self._buf: Deque[LatentState] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def push(self, levels: Sequence[torch.Tensor], unix_seconds: float) -> None:
        self._buf.append(
            LatentState(
                levels=[lv.detach().clone() for lv in levels],
                unix_seconds=float(unix_seconds),
            )
        )

    def history(self, k: int) -> Optional[List[torch.Tensor]]:
        """The most recent ``k`` states stacked on a new leading time axis, one
        tensor per pyramid level: ``(k, n_i, c_i, nlev)``, oldest first.

        Returns ``None`` until ``k`` states have accumulated -- the caller skips
        the step rather than training on a short, zero-padded history.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if len(self._buf) < k:
            return None
        window = list(self._buf)[-k:]
        return [
            torch.stack([state.levels[i] for state in window], dim=0)
            for i in range(len(window[0].levels))
        ]

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> List[LatentState]:
        """Uniform sample of ``n`` past states, without replacement."""
        if not self.allow_replay:
            raise RuntimeError(
                "replay requested while allow_replay=False: sampling a buffer whose "
                "latents were produced by a still-training encoder is unsound "
                "(staleness). Freeze the encoder and set allow_replay=True first."
            )
        if n > len(self._buf):
            raise ValueError(f"asked for {n} samples, buffer holds {len(self._buf)}")
        idx = torch.randperm(len(self._buf), generator=generator)[:n].tolist()
        return [self._buf[i] for i in idx]

    def numel(self) -> int:
        """Total stored elements -- for reporting the memory the cache actually
        costs, which is the whole point of compressing before caching."""
        return sum(state.numel() for state in self._buf)
