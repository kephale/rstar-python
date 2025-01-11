import pytest
from rstar_python import PyRTree
import numpy as np

def test_create_rtree():
    tree = PyRTree(dims=3)
    assert tree.size() == 0

def test_invalid_dimensions():
    with pytest.raises(ValueError):
        PyRTree(dims=5)
    with pytest.raises(ValueError):
        PyRTree(dims=0)

def test_insert_and_size():
    tree = PyRTree(dims=2)
    tree.insert([1.0, 2.0])
    assert tree.size() == 1
    tree.insert([3.0, 4.0])
    assert tree.size() == 2

def test_nearest_neighbor():
    tree = PyRTree(dims=2)
    points = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0]
    ]
    for point in points:
        tree.insert(point)
    
    nearest = tree.nearest_neighbor([0.1, 0.1])
    assert nearest is not None
    assert np.allclose(nearest, [0.0, 0.0])

def test_k_nearest_neighbors():
    tree = PyRTree(dims=2)
    points = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0]
    ]
    for point in points:
        tree.insert(point)
    
    k_nearest = tree.k_nearest_neighbors([0.1, 0.1], k=2)
    assert len(k_nearest) == 2
    assert np.allclose(k_nearest[0], [0.0, 0.0])
    assert np.allclose(k_nearest[1], [1.0, 1.0])

def test_neighbors_within_radius():
    tree = PyRTree(dims=2)
    points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0]
    ]
    for point in points:
        tree.insert(point)
    
    neighbors = tree.neighbors_within_radius([0.0, 0.0], radius=1.5)
    assert len(neighbors) == 2
    assert any(np.allclose(n, [0.0, 0.0]) for n in neighbors)
    assert any(np.allclose(n, [1.0, 0.0]) for n in neighbors)