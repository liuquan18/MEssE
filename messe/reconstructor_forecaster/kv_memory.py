"""kNN key/value memory of one local attention block (Wu et al. 2022,
Memorizing Transformers, `.claude/MEMORIZING TRANSFORMERS.pdf`).

The block attends locally: each backbone cell (token) sends one query against
the keys of its 13-cell window, itself first, then its 12 neighbours
(FieldSpaceNN ``seq_len_zoom`` 0 + ``seq_overlap_space``, ``tokens_with_ring``).
There is one time token, so a cell's key and value describe its whole window
of time slots.

After each update the block's own-cell keys and values of the live window are
stored: one entry per sample step, holding every cell and head, stamped with
the window's last context time. The entries are not re-encoded and carry no
gradient, so they keep the weights of the update that wrote them.

A window reads the entries stamped within ``span`` seconds before its first
context step: never its own context steps, never its future. Each cell
searches the entries of the cells of its window, ``topk`` per head, by cosine
similarity: queries and keys are L2-normalized (the paper's answer to stale
keys, section 3.2) and scaled by a learned factor per head. It attends over
them (V_m) and mixes the result with the block's local attention (V_c) through
one gate per head, as in the paper's eq. 1-2: g·V_m + (1 - g)·V_c,
g = sigmoid(gate). The search runs without gradient; only the chosen entries
are scored again with it, so the backward pass keeps topk keys per query, not
the whole memory. A window without a visible entry gets V_c alone, exactly the
block without memory.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from fieldspacenn.src.modules.transformer.transformer_base import safe_scaled_dot_product_attention


class KVMemory:
    """The last ``size`` stored steps, oldest overwritten first. The first
    ``warmup`` steps offered to :meth:`write` are not stored. ``window``:
    (cells, w) the cells of each cell's attention window, itself first; a
    missing neighbour repeats the cell itself and is not searched."""

    def __init__(self, size: int, span: float, topk: int, window: torch.Tensor, warmup: int = 0):
        self.size, self.span, self.topk, self.warmup = size, span, topk, warmup
        self.window = window
        self.searched = torch.ones_like(window, dtype=torch.bool)
        self.searched[:, 1:] = window[:, 1:] != window[:, :1]
        self.keys = self.values = None  # (size, heads, cells, channels), allocated at the first write
        self.seconds = None  # (size,) stamps; NaN for an empty slot
        self.n_offered = 0
        self._next = 0

    def write(self, k: torch.Tensor, v: torch.Tensor, seconds: float) -> None:
        """Store the own-cell keys and values (heads, cells, channels) of the
        window whose last context step is at ``seconds``. Keys are stored
        normalized."""
        self.n_offered += 1
        if self.n_offered <= self.warmup:
            return
        if self.keys is None:
            self.keys = k.new_zeros((self.size, *k.shape))
            self.values = v.new_zeros((self.size, *v.shape))
            self.seconds = torch.full((self.size,), float("nan"), dtype=torch.float64, device=k.device)
        self.keys[self._next], self.values[self._next] = F.normalize(k.detach(), dim=-1), v.detach()
        self.seconds[self._next] = seconds
        self._next = (self._next + 1) % self.size

    def visible(self, first_seconds: float) -> torch.Tensor:
        """Slots a window whose first context step is at ``first_seconds``
        may read (NaN, an empty slot, compares false)."""
        return (self.seconds < first_seconds) & (self.seconds >= first_seconds - self.span)

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        times: Sequence[Sequence[float]],
        gate: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The block's attention with memory. ``q``: (windows · cells, heads,
        1, channels), ``k``, ``v``: (windows · cells, heads, w, channels),
        FieldSpaceNN's local layout; ``times``: the valid times of each window,
        context steps then forecast; ``gate``, ``log_scale``: (heads,).
        Returns the output like ``q`` and, for the log, the age in hours of
        window 0's retrieved entries before its forecast time,
        attention-weighted, per (head, cell); ``None`` if window 0 has no
        visible entry."""
        local = safe_scaled_dot_product_attention(q, k, v)
        windows, (cells, w), heads = len(times), self.window.shape, q.shape[1]
        local_cells = local.view(windows, cells, heads, -1)
        g = torch.sigmoid(gate)[None, :, None]
        scale = log_scale.exp()[None, :, None]
        rows, age = [], None
        for b, row in enumerate(times):
            slots = torch.zeros(0, dtype=torch.long) if self.keys is None else torch.nonzero(self.visible(row[0]))[:, 0]
            if not len(slots):  # the gate and scale stay in the graph, with zero gradient (DDP)
                rows.append(local_cells[b] + 0.0 * (g + scale).sum())
                continue
            query = F.normalize(q.view(windows, cells, heads, -1)[b], dim=-1)  # (cells, heads, channels)
            keys, values = self.keys[slots], self.values[slots]  # (slots, heads, cells, channels)
            with torch.no_grad():  # cosine of every visible entry of every window cell, (cells, heads, slots, w)
                scores = torch.stack(
                    [torch.einsum("nhc,shnc->nhs", query, keys[:, :, self.window[:, j]]) for j in range(w)], dim=-1
                )
                scores.masked_fill_(~self.searched[:, None, None, :], float("-inf"))
                pick = scores.flatten(2).topk(min(self.topk, len(slots) * w), dim=-1).indices  # (cells, heads, topk)
            slot, cell = pick // w, self.window.gather(1, (pick % w).flatten(1)).view(pick.shape)
            head = torch.arange(heads, device=q.device)[None, :, None]
            chosen = torch.isfinite(scores.flatten(2).gather(-1, pick))  # fewer searched entries than topk
            logits = scale * torch.einsum("nhc,nhkc->nhk", query, keys[slot, head, cell])
            weights = logits.masked_fill(~chosen, float("-inf")).softmax(dim=-1)
            memory = torch.einsum("nhk,nhkc->nhc", weights, values[slot, head, cell])
            rows.append(g * memory + (1 - g) * local_cells[b])
            if b == 0:
                hours = ((row[-1] - self.seconds[slots]) / 3600.0).to(weights.dtype)
                age = (weights.detach() * hours[slot]).sum(dim=-1).T  # (heads, cells)
        return torch.stack(rows).view_as(local), age
