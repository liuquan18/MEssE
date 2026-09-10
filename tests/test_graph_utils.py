"""Unit tests for graph_utils.build_local_graph, using a synthetic
stand-in for COMIN's `domain.cells` descriptive data instead of a live
ICON simulation.

Layout under test: nproma=4, one block (nblks=1) -> 4 local node slots.
Three real triangular cells (0, 1, 2) mutually adjacent (a closed triangle)
plus one unused padding slot (3, since nproma doesn't divide the real cell
count evenly — the common case at the end of the last block). Cell 0 and 1
are owned (prognostic); cell 2 is a halo cell mirrored from a neighboring
rank; cell 3 is unused padding.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graph_utils import build_local_graph


def _make_domain():
    nproma, nblks, n_nbr = 4, 1, 3

    # 1-based neighbor (idx, blk); (0, 0) marks "no neighbor".
    # cell 0 (idx=1) <-> cell 1 (idx=2), cell 0 <-> cell 2 (idx=3)
    # cell 1 (idx=2) <-> cell 0, cell 1 <-> cell 2
    # cell 2 (idx=3) <-> cell 0, cell 2 <-> cell 1
    # cell 3 (idx=4, padding): no neighbors
    neighbor_idx = np.array(
        [
            [[2, 3, 0]],
            [[1, 3, 0]],
            [[1, 2, 0]],
            [[0, 0, 0]],
        ],
        dtype=np.int64,
    ).reshape(nproma, nblks, n_nbr)
    neighbor_blk = np.array(
        [
            [[1, 1, 0]],
            [[1, 1, 0]],
            [[1, 1, 0]],
            [[0, 0, 0]],
        ],
        dtype=np.int64,
    ).reshape(nproma, nblks, n_nbr)

    clon = np.array([0.0, 0.1, 0.2, 0.0])
    clat = np.array([0.0, 0.0, 0.0, 0.0])
    # decomp_domain: 0 = owned, >0 = halo, <0 = unused padding.
    decomp_domain = np.array([0, 0, 1, -1])

    cells = SimpleNamespace(
        ncells=3,  # owned + halo (excludes the padding slot)
        nblks=nblks,
        clon=clon,
        clat=clat,
        decomp_domain=decomp_domain,
        neighbor_idx=neighbor_idx,
        neighbor_blk=neighbor_blk,
    )
    return SimpleNamespace(cells=cells)


def test_build_local_graph_shapes_and_ownership():
    domain = _make_domain()
    graph = build_local_graph(domain, nproma=4, device=torch.device("cpu"))

    assert graph.n_nodes == 4  # nproma * nblks
    assert graph.n_valid == 3  # domain.cells.ncells
    assert graph.owned_mask.tolist() == [True, True, False, False]


def test_build_local_graph_edges_are_symmetric_with_self_loops():
    domain = _make_domain()
    graph = build_local_graph(domain, nproma=4, device=torch.device("cpu"), add_self_loops=True)

    edges = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))

    # Triangle among nodes 0, 1, 2, both directions.
    triangle = {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)}
    assert triangle <= edges

    # Self-loops on every node, including the isolated padding node 3.
    for i in range(4):
        assert (i, i) in edges

    # Node 3 (padding, no real neighbors) only has its self-loop.
    edges_touching_3 = {e for e in edges if 3 in e}
    assert edges_touching_3 == {(3, 3)}

    # No duplicate edges: exactly the 6 triangle edges + 4 self-loops.
    assert graph.edge_index.shape[1] == 10
    assert len(edges) == 10


def test_build_local_graph_without_self_loops():
    domain = _make_domain()
    graph = build_local_graph(domain, nproma=4, device=torch.device("cpu"), add_self_loops=False)

    edges = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
    assert edges == {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)}


def test_build_local_graph_edge_attr_zero_for_self_loops():
    domain = _make_domain()
    graph = build_local_graph(domain, nproma=4, device=torch.device("cpu"))

    for src, dst, attr in zip(
        graph.edge_index[0].tolist(), graph.edge_index[1].tolist(), graph.edge_attr.tolist()
    ):
        if src == dst:
            assert attr == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_build_local_graph_moves_tensors_to_device():
    domain = _make_domain()
    graph = build_local_graph(domain, nproma=4, device=torch.device("cpu"))
    for t in (graph.edge_index, graph.owned_mask, graph.clon, graph.clat, graph.edge_attr):
        assert t.device.type == "cpu"
