"""Unit tests for MEssE.utils.icon_online_helper: the generic,
YAC/HEALPix-independent COMIN<->tensor/MPI helpers, online normalization
statistics, and per-timestep training-example scheduling shared across
online-training plugins. Pure numpy/torch — no comin, no MPI, no GPU, no
live ICON simulation required.
"""

import datetime

import numpy as np
import pytest
import torch

from MEssE.utils.icon_online_helper import (
    RolloutBuffer,
    RunningMeanStd,
    extract_icon_cells,
    insert_icon_cells,
    parse_icon_datetime,
)


# ----------------------------------------------------------------------------
# parse_icon_datetime
# ----------------------------------------------------------------------------


def test_parse_icon_datetime():
    dt = parse_icon_datetime("2026-09-10T12:34:56.789")
    assert dt == datetime.datetime(2026, 9, 10, 12, 34, 56, tzinfo=datetime.timezone.utc)


def test_parse_icon_datetime_no_fractional_seconds():
    dt = parse_icon_datetime("2026-01-01T00:00:00")
    assert dt == datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)


# ----------------------------------------------------------------------------
# extract_icon_cells / insert_icon_cells
# ----------------------------------------------------------------------------


def _make_comin_buffer(nproma, nlev, nblk, seed=0):
    """A COMIN-shaped (nproma, nlev, nblk) buffer with distinct values per
    cell so extraction order bugs would show up as wrong values, not just
    wrong shapes."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nproma, nlev, nblk))


def test_extract_icon_cells_3d_matches_fortran_unravel():
    nproma, nlev, nblk = 4, 3, 2
    buf = _make_comin_buffer(nproma, nlev, nblk)
    nc = nproma * nblk  # no padding cells in this case

    extracted = extract_icon_cells(buf, nc)
    assert extracted.shape == (nc, nlev)

    # Cell c (0-based, Fortran order) is buf[c % nproma, :, c // nproma].
    for c in range(nc):
        expected = buf[c % nproma, :, c // nproma]
        assert np.allclose(extracted[c], expected)


def test_extract_icon_cells_drops_padding_beyond_nc():
    nproma, nlev, nblk = 4, 2, 1
    buf = _make_comin_buffer(nproma, nlev, nblk)
    nc = 3  # only 3 of the 4 slots are real (owned+halo) cells

    extracted = extract_icon_cells(buf, nc)
    assert extracted.shape == (nc, nlev)


def test_extract_icon_cells_2d_surface_field():
    nproma, nblk = 4, 2
    buf = _make_comin_buffer(nproma, 1, nblk)[:, 0, :]  # (nproma, nblk), single level
    nc = nproma * nblk

    extracted = extract_icon_cells(buf, nc)
    assert extracted.shape == (nc,)
    for c in range(nc):
        assert extracted[c] == pytest.approx(buf[c % nproma, c // nproma])


def test_extract_icon_cells_drops_trailing_singleton_dims():
    nproma, nlev, nblk = 4, 3, 2
    buf = _make_comin_buffer(nproma, nlev, nblk)[..., None]  # COMIN sometimes adds a (..., 1) axis
    nc = nproma * nblk

    extracted = extract_icon_cells(buf, nc)
    assert extracted.shape == (nc, nlev)


def test_insert_icon_cells_inverts_extract_icon_cells():
    nproma, nlev, nblk = 4, 3, 2
    buf = _make_comin_buffer(nproma, nlev, nblk)
    nc = nproma * nblk

    extracted = extract_icon_cells(buf, nc)

    target = np.zeros_like(buf)
    insert_icon_cells(np.asarray(extracted), target)
    assert np.allclose(target, buf)


def test_insert_icon_cells_respects_indices():
    nproma, nlev, nblk = 4, 2, 1
    buf = np.zeros((nproma, nlev, nblk))
    # Only write cells 0 and 2 (e.g. "owned" cells in a graph with halos).
    indices = np.array([0, 2])
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    insert_icon_cells(values, buf, indices=indices)

    assert np.allclose(buf[0, :, 0], [1.0, 2.0])
    assert np.allclose(buf[2, :, 0], [3.0, 4.0])
    # Untouched cells stay zero.
    assert np.allclose(buf[1, :, 0], [0.0, 0.0])
    assert np.allclose(buf[3, :, 0], [0.0, 0.0])


# ----------------------------------------------------------------------------
# RunningMeanStd
# ----------------------------------------------------------------------------


def test_running_mean_std_matches_torch_moments():
    torch.manual_seed(0)
    n_samples, shape = 50, (6, 3)
    samples = torch.randn(n_samples, *shape) * 2.0 + 5.0

    acc = RunningMeanStd(shape=shape, n_samples=n_samples, device=torch.device("cpu"))
    for i in range(n_samples):
        done = acc.update(samples[i])
        assert done == (i == n_samples - 1)

    assert acc.done
    expected_mean = samples.mean(dim=0)
    expected_std = samples.std(dim=0, unbiased=False)
    assert torch.allclose(acc.stats.mean, expected_mean, atol=1e-4)
    assert torch.allclose(acc.stats.std, expected_std, atol=1e-4)


def test_running_mean_std_freezes_after_n_samples():
    acc = RunningMeanStd(shape=(2,), n_samples=3, device=torch.device("cpu"))
    acc.update(torch.tensor([1.0, 1.0]))
    acc.update(torch.tensor([2.0, 2.0]))
    assert not acc.done
    acc.update(torch.tensor([3.0, 3.0]))
    assert acc.done
    frozen_mean = acc.stats.mean.clone()

    # Further updates must be no-ops once frozen.
    assert acc.update(torch.tensor([1000.0, -1000.0])) is True
    assert torch.equal(acc.stats.mean, frozen_mean)
    assert acc.count == 3


def test_running_mean_std_normalize_denormalize_roundtrip():
    acc = RunningMeanStd(shape=(4,), n_samples=2, device=torch.device("cpu"))
    acc.update(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    acc.update(torch.tensor([3.0, 4.0, 5.0, 6.0]))
    assert acc.done

    x = torch.tensor([2.5, 3.5, 4.5, 5.5])
    normalized = acc.normalize(x)
    x_back = acc.denormalize(normalized)
    assert torch.allclose(x_back, x, atol=1e-5)


def test_running_mean_std_raises_before_finalized():
    acc = RunningMeanStd(shape=(2,), n_samples=5, device=torch.device("cpu"))
    acc.update(torch.zeros(2))
    with pytest.raises(AssertionError):
        acc.normalize(torch.zeros(2))


# ----------------------------------------------------------------------------
# RolloutBuffer
# ----------------------------------------------------------------------------


def test_rollout_buffer_fixed_horizon_and_slides():
    buf = RolloutBuffer(n_steps=3)

    assert buf.push("s0") is None  # becomes source
    assert buf.push("t1") is None  # 1 step after source
    assert buf.push("t2") is None  # 2 steps after source
    ready = buf.push("t3")  # 3 steps after source -> horizon reached
    assert ready == ("s0", "t3")
    # Snapshots pushed in between (t1, t2) were never returned or compared.

    # Horizon slides forward: "t3" becomes the next source.
    assert buf.push("t4") is None
    assert buf.push("t5") is None
    ready2 = buf.push("t6")
    assert ready2 == ("t3", "t6")


def test_rollout_buffer_reset():
    buf = RolloutBuffer(n_steps=2)
    buf.push("s0")
    buf.push("t1")
    assert buf.source is not None
    assert buf._pushes_since_source == 1

    buf.reset()
    assert buf.source is None
    assert buf._pushes_since_source == 0
    # After reset, the buffer behaves like a fresh one.
    assert buf.push("s0") is None
    assert buf.push("t1") is None
    assert buf.push("t2") == ("s0", "t2")


def test_rollout_buffer_single_step_horizon():
    """n_steps=1 is a degenerate but valid case: every push after the first
    immediately returns a ready (source, target) pair."""
    buf = RolloutBuffer(n_steps=1)
    assert buf.push("s0") is None
    assert buf.push("t1") == ("s0", "t1")
    assert buf.push("t2") == ("t1", "t2")
