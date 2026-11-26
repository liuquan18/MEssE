"""Unit tests for messe.common.icon_data: ComIn arrays <-> flat cell arrays and ICON's
clock. Pure numpy/torch: no comin, no MPI, no GPU.
"""

import datetime

import numpy as np
import pytest

from messe.common_utils.icon_data import extract_icon_cells, insert_icon_cells, parse_icon_datetime, sample_interval


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


# --------------------------------------------------------------------------
# sample_interval
# --------------------------------------------------------------------------


def test_sample_interval_converts_seconds_to_whole_steps():
    assert sample_interval(600, 60) == (10, 600)     # R2B8 today: 60 s steps
    assert sample_interval(600, 10) == (60, 600)     # same 10 minutes at 10 s steps
    assert sample_interval(600, 450) == (1, 450)     # 450 s steps: nearest is one step
    assert sample_interval(1800, 450) == (4, 1800)


def test_sample_interval_rounds_to_the_realizable_interval():
    stride, interval = sample_interval(600, 45)      # 13.3 steps -> 13
    assert (stride, interval) == (13, 585)


def test_sample_interval_is_at_least_one_step_and_needs_a_positive_timestep():
    assert sample_interval(5, 60) == (1, 60)         # asked for less than one step
    with pytest.raises(ValueError):
        sample_interval(600, 0)
