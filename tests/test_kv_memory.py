"""Tests for kv_memory.py on synthetic keys and values (CPU, no model), in
FieldSpaceNN's local attention layout: one query per cell against the keys of
its window, the cell itself first."""

import math

import pytest
import torch
import torch.nn.functional as F

from fieldspacenn.src.modules.transformer.transformer_base import safe_scaled_dot_product_attention
from kv_memory import KVMemory

HEADS, CELLS, CHANNELS = 2, 5, 4
# Each cell's window: itself, then its neighbours. Cell 4 misses one neighbour (repeated self, not searched).
WINDOW = torch.tensor([[0, 1, 2], [1, 0, 3], [2, 0, 4], [3, 1, 4], [4, 2, 4]])
W = WINDOW.shape[1]
T0, DT = 1.78e9, 600.0
SCALE = torch.full((HEADS,), math.log(10.0))


def _local(seed, windows=1):
    """q (windows·cells, heads, 1, ch) and k, v (windows·cells, heads, W, ch)."""
    torch.manual_seed(seed)
    q = torch.randn(windows * CELLS, HEADS, 1, CHANNELS)
    k, v = (torch.randn(windows * CELLS, HEADS, W, CHANNELS) for _ in range(2))
    return q, k, v


def _entry(step):
    """Own-cell keys and values (heads, cells, ch) stored at ``step``."""
    torch.manual_seed(1000 + step)
    return torch.randn(HEADS, CELLS, CHANNELS), torch.randn(HEADS, CELLS, CHANNELS)


def _times(first_step, n_context=2):
    """Valid times of a window whose first context step is ``first_step``."""
    return [T0 + (first_step + j) * DT for j in range(n_context + 1)]


def _filled(n_steps, size=100, span=1e9, topk=3, warmup=0):
    """A memory with entries stamped at steps 0 .. n_steps - 1."""
    memory = KVMemory(size, span, topk, WINDOW, warmup)
    for step in range(n_steps):
        memory.write(*_entry(step), T0 + step * DT)
    return memory


def test_ring_keeps_the_last_steps_skips_the_warmup_and_stores_normalized_keys():
    memory = _filled(9, size=4, warmup=2)
    assert memory.n_offered == 9
    stamps = sorted(((memory.seconds - T0) / DT).round().long().tolist())
    assert stamps == [5, 6, 7, 8]
    newest = torch.argmax(memory.seconds)
    k, v = _entry(8)
    assert torch.allclose(memory.keys[newest], F.normalize(k, dim=-1)) and torch.equal(memory.values[newest], v)
    assert not memory.keys.requires_grad
    assert memory.searched.tolist() == [[True] * 3] * 4 + [[True, True, False]]


def test_without_visible_entries_the_output_is_the_local_attention_exactly():
    q, k, v = _local(50, windows=2)
    local = safe_scaled_dot_product_attention(q, k, v)
    gate = torch.zeros(HEADS)
    out, age = KVMemory(10, 1e9, 3, WINDOW).attend(q, k, v, [_times(0), _times(1)], gate, SCALE)  # never written
    assert torch.equal(out, local) and age is None
    memory = _filled(3)  # stamps 0-2: not before a window starting at step 0
    out, age = memory.attend(q, k, v, [_times(0), _times(-5)], gate, SCALE)
    assert torch.equal(out, local) and age is None


def test_a_window_reads_only_stamps_within_the_span_before_its_first_context_step():
    """Entries stamped at the first context step or later, or older than the span, get no weight even
    with a perfect key; with one visible entry and topk 1 the memory output is that entry's value of
    the best-matching window cell."""
    q, k, v = _local(60)
    memory = KVMemory(10, span=3 * DT, topk=1, window=WINDOW)
    first = 10
    query = q[:, :, 0].transpose(0, 1)  # (heads, cells, ch)
    visible_value = torch.randn(HEADS, CELLS, CHANNELS)
    for step in (first - 4, first - 2, first, first + 1):  # too old, visible, own context, future
        value = visible_value if step == first - 2 else torch.full((HEADS, CELLS, CHANNELS), 1e3)
        memory.write(query, value, T0 + step * DT)  # each cell's key = its own query
    gate = torch.tensor([20.0, -20.0])  # head 0 reads the memory only, head 1 not at all
    out, age = memory.attend(q, k, v, [_times(first)], gate, SCALE)
    local = safe_scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out[:, 0, 0], visible_value[0], atol=1e-6)  # cosine 1 with its own entry
    assert torch.allclose(out[:, 1], local[:, 1], atol=1e-6)
    hours_before_forecast = (_times(first)[-1] - (T0 + (first - 2) * DT)) / 3600
    assert torch.allclose(age, torch.full((HEADS, CELLS), hours_before_forecast))


def test_topk_attention_matches_a_brute_force_search_over_the_window():
    memory = _filled(12, topk=4)
    q, k, v = _local(70, windows=2)
    gate, log_scale = torch.tensor([0.3, -0.7]), torch.tensor([1.5, 2.5])
    times = [_times(20), _times(8)]  # all 12 entries visible, and only stamps 0-7
    out, _ = memory.attend(q, k, v, times, gate, log_scale)
    local = safe_scaled_dot_product_attention(q, k, v)
    g = torch.sigmoid(gate)
    for b, row in enumerate(times):
        steps = [s for s in range(12) if T0 + s * DT < row[0]]
        for h in range(HEADS):
            for cell in range(CELLS):
                neighbours = [c for j, c in enumerate(WINDOW[cell].tolist()) if j == 0 or c != cell]
                keys = torch.stack([_entry(s)[0][h, c] for s in steps for c in neighbours])
                values = torch.stack([_entry(s)[1][h, c] for s in steps for c in neighbours])
                query = q[b * CELLS + cell, h, 0]
                cosine = F.normalize(keys, dim=-1) @ F.normalize(query, dim=0)
                best = cosine.topk(4).indices
                expected = (log_scale[h].exp() * cosine[best]).softmax(0) @ values[best]
                mixed = g[h] * expected + (1 - g[h]) * local[b * CELLS + cell, h, 0]
                assert torch.allclose(out[b * CELLS + cell, h, 0], mixed, atol=1e-5), (b, h, cell)


def test_each_cell_reads_only_the_entries_of_its_window():
    memory = _filled(6)
    q, k, v = _local(80)
    gate = torch.zeros(HEADS)
    before, _ = memory.attend(q, k, v, [_times(10)], gate, SCALE)
    memory.keys[:, :, 3] = F.normalize(torch.randn_like(memory.keys[:, :, 3]), dim=-1)  # cell 3's entries
    memory.values[:, :, 3] = torch.randn_like(memory.values[:, :, 3])
    after, _ = memory.attend(q, k, v, [_times(10)], gate, SCALE)
    reach = [c for c in range(CELLS) if 3 in WINDOW[c].tolist()]  # cells 1 and 3
    assert reach == [1, 3]
    assert all(torch.equal(before[c], after[c]) for c in range(CELLS) if c not in reach)
    assert all(not torch.equal(before[c], after[c]) for c in reach)


def test_stored_key_scale_does_not_change_the_result():
    q, k, v = _local(85)
    gate = torch.zeros(HEADS)
    outs = []
    for factor in (1.0, 50.0):
        memory = KVMemory(100, 1e9, 3, WINDOW)
        for step in range(6):
            key, value = _entry(step)
            memory.write(key * factor, value, T0 + step * DT)
        outs.append(memory.attend(q, k, v, [_times(10)], gate, SCALE)[0])
    assert torch.allclose(*outs, atol=1e-6)


def test_gradients_reach_the_queries_gate_and_scale_but_not_the_stored_entries():
    memory = _filled(6)
    q, k, v = (t.requires_grad_() for t in _local(90))
    gate, log_scale = torch.zeros(HEADS, requires_grad=True), SCALE.clone().requires_grad_()
    out, _ = memory.attend(q, k, v, [_times(10)], gate, log_scale)
    out.square().sum().backward()
    assert q.grad.abs().sum() > 0 and gate.grad.abs().sum() > 0 and log_scale.grad.abs().sum() > 0
    assert memory.keys.grad is None and not memory.keys.requires_grad and not memory.values.requires_grad


@pytest.mark.parametrize("written", [0, 3])
def test_without_visible_entries_the_gate_and_scale_still_get_a_zero_gradient(written):
    """DDP (find_unused_parameters=False) needs a gradient for every parameter."""
    memory = _filled(written)
    q, k, v = _local(95)
    gate, log_scale = torch.zeros(HEADS, requires_grad=True), SCALE.clone().requires_grad_()
    out, _ = memory.attend(q, k, v, [_times(0)], gate, log_scale)
    out.sum().backward()
    assert torch.equal(gate.grad, torch.zeros(HEADS)) and torch.equal(log_scale.grad, torch.zeros(HEADS))
