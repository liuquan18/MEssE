"""Forcings the RF plugin computes itself instead of reading them from ICON:
TOA insolation from the model time, and the cell position on the sphere.
"""

import datetime

import numpy as np
import torch


def toa_insolation(unix_seconds: float, lon: torch.Tensor, lat: torch.Tensor, tsi: float = 1361.0) -> torch.Tensor:
    """Incoming shortwave flux at the top of the atmosphere [W/m2] at UTC time
    ``unix_seconds`` for cell centres ``lon``/``lat`` [radian]:
    ``tsi * (r0/r)^2 * max(0, cos(solar zenith angle))``, the form of ICON's
    AES ``rsdt = rsdt0 * cosmu0``. Declination, equation of time and
    Earth-Sun distance from Spencer (1971), accurate to a few W/m2."""
    time = datetime.datetime.fromtimestamp(unix_seconds, tz=datetime.timezone.utc)
    start = datetime.datetime(time.year, 1, 1, tzinfo=time.tzinfo)
    year_days = (datetime.datetime(time.year + 1, 1, 1, tzinfo=time.tzinfo) - start).days
    day = (time - start).total_seconds() / 86400.0  # fractional day of the year, 0 at Jan 1 00 UTC
    g = 2 * np.pi * day / year_days
    declination = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
                   + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    equation_of_time = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                                 - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))  # minutes
    distance_factor = (1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g)
                       + 0.000719 * np.cos(2 * g) + 0.000077 * np.sin(2 * g))  # (r0/r)^2
    utc_minutes = (day % 1.0) * 1440.0
    hour_angle = (utc_minutes + equation_of_time) / 1440.0 * 2 * np.pi + lon - np.pi
    cos_zenith = torch.sin(lat) * np.sin(declination) + torch.cos(lat) * np.cos(declination) * torch.cos(hour_angle)
    return tsi * distance_factor * cos_zenith.clamp(min=0)


def cell_position(lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
    """Cell centres ``lon``/``lat`` [radian] as unit vectors on the sphere,
    (n, 3) = (cos lat cos lon, cos lat sin lon, sin lat): position without the
    jump of longitude at the dateline or its singularity at the poles."""
    return torch.stack([torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)], dim=1)
