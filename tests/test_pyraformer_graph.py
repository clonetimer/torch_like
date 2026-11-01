"""Literal source-geometry fixtures catch lost tails, wrong offsets and dense storage."""

import importlib
import importlib.util

import pytest
import torch


def layers():
    name = "torch_like.models.forecasting.pyraformer_layers"
    assert importlib.util.find_spec(name) is not None, "Pyraformer sparse operators are missing"
    return importlib.import_module(name)


def dense_graph(lengths, factors, inner_size):
    """Independent interval-filling oracle; never calls the production builder."""
    total = sum(lengths)
    adjacency = torch.zeros(total, total, dtype=torch.bool)
    offset = 0
    for level, size in enumerate(lengths):
        for i in range(size):
            lo, hi = max(0, i - inner_size // 2), min(size, i + inner_size // 2 + 1)
            adjacency[offset + i, offset + lo : offset + hi] = True
        if level:
            child_offset = offset - lengths[level - 1]
            factor = factors[level - 1]
            for parent in range(size):
                start = child_offset + factor * parent
                stop = child_offset + (factor * (parent + 1) if parent < size - 1 else lengths[level - 1])
                adjacency[offset + parent, start:stop] = True
                adjacency[start:stop, offset + parent] = True
        offset += size
    return adjacency


def test_literal_three_level_ragged_graph():
    sizes, neighbors, valid, ancestors = layers()._pyramid_graph(11, (3, 2), 3)
    assert sizes == (11, 3, 1)
    expected = [
        [0, 1, 11],
        [0, 1, 2, 11],
        [1, 2, 3, 11],
        [2, 3, 4, 12],
        [3, 4, 5, 12],
        [4, 5, 6, 12],
        [5, 6, 7, 13],
        [6, 7, 8, 13],
        [7, 8, 9, 13],
        [8, 9, 10, 13],
        [9, 10, 13],
        [0, 1, 2, 11, 12, 14],
        [3, 4, 5, 11, 12, 13, 14],
        [6, 7, 8, 9, 10, 12, 13, 14],
        [11, 12, 13, 14],
    ]
    assert [row[mask].tolist() for row, mask in zip(neighbors, valid, strict=True)] == expected
    assert ancestors.tolist() == [10, 13, 14]


def test_literal_four_level_remainder_ownership():
    sizes, neighbors, valid, ancestors = layers()._pyramid_graph(23, (3, 2, 2), 3)
    assert sizes == (23, 7, 3, 1)
    rows = {
        0: [0, 1, 23],
        20: [19, 20, 21, 29],
        21: [20, 21, 22, 29],
        22: [21, 22, 29],
        23: [0, 1, 2, 23, 24, 30],
        28: [15, 16, 17, 27, 28, 29, 32],
        29: [18, 19, 20, 21, 22, 28, 29, 32],
        30: [23, 24, 30, 31, 33],
        31: [25, 26, 30, 31, 32, 33],
        32: [27, 28, 29, 31, 32, 33],
        33: [30, 31, 32, 33],
    }
    for node, expected in rows.items():
        assert neighbors[node][valid[node]].tolist() == expected
    assert ancestors.tolist() == [22, 29, 32, 33]


@pytest.mark.parametrize("length,factors,inner", [(8, (2, 2, 2), 1), (23, (3, 2, 2), 3), (97, (4, 3, 2), 5)])
def test_graph_matches_independent_intervals_and_has_unique_symmetric_edges(length, factors, inner):
    sizes, neighbors, valid, ancestors = layers()._pyramid_graph(length, factors, inner)
    adjacency = dense_graph(sizes, factors, inner)
    actual = torch.zeros_like(adjacency)
    for node, (row, mask) in enumerate(zip(neighbors, valid, strict=True)):
        keys = row[mask]
        assert len(keys.unique()) == len(keys)
        actual[node, keys] = True
    assert torch.equal(actual, adjacency)
    assert torch.equal(actual, actual.T)
    assert actual.diag().all()
    assert neighbors.dtype == ancestors.dtype == torch.long
    assert valid.dtype == torch.bool


def test_topology_storage_scales_linearly_not_as_a_dense_square():
    for fine in (65, 257, 1025, 4097):
        sizes, neighbors, valid, _ = layers()._pyramid_graph(fine, (4, 4, 4), 3)
        assert neighbors.shape == valid.shape
        assert neighbors.shape[0] == sum(sizes)
        assert neighbors.shape[1] <= 11
        assert neighbors.numel() <= 11 * sum(sizes)
