"""Unit tests for messe.reconstructor_forecaster.forcings: TOA insolation and cell position."""

import datetime

import numpy as np
import torch

from messe.reconstructor_forecaster.forcings import cell_position, toa_insolation


# ----------------------------------------------------------------------------
# toa_insolation
# ----------------------------------------------------------------------------


def test_toa_insolation_geometry():
    def at(when, lon_deg, lat_deg):
        seconds = datetime.datetime(*when, tzinfo=datetime.timezone.utc).timestamp()
        return float(toa_insolation(seconds, torch.tensor([np.deg2rad(lon_deg)]), torch.tensor([np.deg2rad(lat_deg)]))[0])

    noon_equinox = (2020, 3, 20, 12, 0)
    assert 1330 < at(noon_equinox, 0, 0) < 1375  # sun near the zenith at 0E, 0N
    assert at(noon_equinox, 180, 0) == 0  # night side
    january = (2020, 1, 3, 6, 0)
    assert at(january, 30, 89.9) == 0  # polar night
    assert 500 < at(january, 30, -89.9) < 560  # polar day: tsi * sin(22.8 deg) * (r0/r)^2


def test_toa_insolation_global_mean_is_a_quarter_of_the_incoming_flux():
    lon, lat = np.meshgrid(np.linspace(-np.pi, np.pi, 721)[:-1], np.linspace(-np.pi / 2, np.pi / 2, 361))
    weights = np.cos(lat)
    for when, distance_factor in [((2020, 1, 3, 17, 0), 1.0342), ((2020, 7, 4, 3, 0), 0.9670)]:
        seconds = datetime.datetime(*when, tzinfo=datetime.timezone.utc).timestamp()
        flux = toa_insolation(seconds, torch.from_numpy(lon), torch.from_numpy(lat)).numpy()
        mean = (flux * weights).sum() / weights.sum()
        assert abs(mean - 1361 * distance_factor / 4) < 2, (when, mean)


# ----------------------------------------------------------------------------
# cell_position
# ----------------------------------------------------------------------------


def test_cell_position_is_a_unit_vector_without_a_dateline_or_pole_jump():
    def at(lon_deg, lat_deg):
        lon, lat = torch.tensor([np.deg2rad(lon_deg)]).float(), torch.tensor([np.deg2rad(lat_deg)]).float()
        return cell_position(lon, lat)[0]

    torch.testing.assert_close(at(0, 0), torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(at(90, 0), torch.tensor([0.0, 1.0, 0.0]), atol=1e-7, rtol=0)
    torch.testing.assert_close(at(0, 90), torch.tensor([0.0, 0.0, 1.0]), atol=1e-7, rtol=0)
    torch.testing.assert_close(at(180, 30), at(-180, 30))  # dateline
    torch.testing.assert_close(at(10, 90), at(-120, 90), atol=1e-7, rtol=0)  # every longitude at the pole

    lon, lat = torch.rand(1000) * 2 * np.pi - np.pi, torch.rand(1000) * np.pi - np.pi / 2
    torch.testing.assert_close(cell_position(lon, lat).norm(dim=1), torch.ones(1000))


def test_cell_position_global_mean_is_zero_and_std_one_over_sqrt3():
    # what the plugin's global normalization of the position columns sees
    lon, lat = np.meshgrid(np.linspace(-np.pi, np.pi, 721)[:-1], np.linspace(-np.pi / 2, np.pi / 2, 361))
    weights = np.cos(lat).ravel()[:, None]
    position = cell_position(torch.from_numpy(lon.ravel()), torch.from_numpy(lat.ravel())).numpy()
    mean = (position * weights).sum(axis=0) / weights.sum()
    std = np.sqrt((position**2 * weights).sum(axis=0) / weights.sum() - mean**2)
    np.testing.assert_allclose(mean, 0, atol=1e-3)
    np.testing.assert_allclose(std, 1 / np.sqrt(3), atol=2e-3)
