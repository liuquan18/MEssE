"""A small icosahedral triangular grid for tests, as an ICON grid file gives
it: nested (the 4 children of a cell in consecutive rows), counter-clockwise
vertices, but with the children and each cell's vertices in a shuffled order."""

import numpy as np
import torch

from fieldspacenn.src.modules.grids.icon_grid import canonical_mgrid


def icosahedral_grid(n_levels: int, seed: int = 0):
    """``(vertex_of_cell (n, 3), vertex_xyz (n_vertices, 3))`` of the
    icosahedron (poles at two vertices) after ``n_levels`` bisections:
    20 * 4**n_levels cells."""
    rng = np.random.default_rng(seed)
    lat = np.arctan(0.5)
    ring = lambda sign, shift: [  # noqa: E731
        [np.cos(lat) * np.cos(a), np.cos(lat) * np.sin(a), sign * np.sin(lat)]
        for a in np.radians(72 * np.arange(5) + shift)
    ]
    vertices = [[0, 0, 1], *ring(1, 0), *ring(-1, 36), [0, 0, -1]]  # north pole, upper ring, lower ring, south pole
    up, low = lambda i: 1 + i % 5, lambda i: 6 + i % 5  # noqa: E731
    cells = []
    for i in range(5):
        cells += [[0, up(i), up(i + 1)], [up(i), low(i), up(i + 1)], [low(i), low(i + 1), up(i + 1)], [11, low(i + 1), low(i)]]

    midpoint = {}

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in midpoint:
            m = np.add(vertices[a], vertices[b])
            vertices.append(list(m / np.linalg.norm(m)))
            midpoint[key] = len(vertices) - 1
        return midpoint[key]

    for _ in range(n_levels):
        children = []
        for a, b, c in cells:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            four = [[ab, bc, ca], [a, ab, ca], [ab, b, bc], [ca, bc, c]]
            children += [list(np.roll(four[k], rng.integers(3))) for k in rng.permutation(4)]
        cells = children
    return np.array(cells), np.array(vertices, dtype=float)


def is_counter_clockwise(vertex_of_cell, vertex_xyz):
    a, b, c = (vertex_xyz[vertex_of_cell[:, k]] for k in range(3))
    return (np.cross(b - a, c - a) * (a + b + c)).sum(-1) > 0


def icosahedral_mgrid(n_zooms: int, n_levels: int, seed: int = 0):
    """`canonical_mgrid` of the test grid: zoom 0 has 20 * 4**(n_levels - n_zooms + 1) cells."""
    return canonical_mgrid(*icosahedral_grid(n_levels, seed), n_zooms)


def unit_vectors(coords: torch.Tensor) -> torch.Tensor:
    """(n, 2) lon, lat -> (n, 3) on the unit sphere."""
    lon, lat = coords.double().T
    return torch.stack([torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)], -1)
