import numpy as np
import pytest

from rstar_python import PyRTree, __version__


def test_version():
    assert isinstance(__version__, str)
    assert __version__.count(".") >= 2


def test_create_rtree():
    tree = PyRTree(dims=3)
    assert tree.size() == 0
    assert len(tree) == 0
    assert tree.dims == 3


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        PyRTree(dims=1)
    with pytest.raises(ValueError):
        PyRTree(dims=9)


@pytest.mark.parametrize("dims", [2, 3, 4, 8])
def test_insert_various_dims(dims):
    tree = PyRTree(dims=dims)
    tree.insert([1.0] * dims)
    tree.insert([2.0] * dims)
    assert tree.size() == 2


def test_insert_wrong_dims_raises():
    tree = PyRTree(dims=2)
    with pytest.raises(ValueError):
        tree.insert([1.0, 2.0, 3.0])


def test_insert_returns_id():
    tree = PyRTree(dims=2)
    assert tree.insert([0.0, 0.0]) == 0
    assert tree.insert([1.0, 1.0]) == 1
    # Explicit id is honoured and advances the auto-counter past it.
    assert tree.insert([2.0, 2.0], data=100) == 100
    assert tree.insert([3.0, 3.0]) == 101


def test_repr_and_contains():
    tree = PyRTree(dims=2)
    tree.insert([1.0, 2.0])
    assert "dims=2" in repr(tree)
    assert "size=1" in repr(tree)
    assert [1.0, 2.0] in tree
    assert [9.0, 9.0] not in tree


def test_nearest_neighbor():
    tree = PyRTree(dims=2)
    for point in [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]:
        tree.insert(point)
    nearest = tree.nearest_neighbor([0.1, 0.1])
    assert nearest is not None
    assert np.allclose(nearest, [0.0, 0.0])


def test_nearest_neighbor_empty():
    assert PyRTree(dims=2).nearest_neighbor([0.0, 0.0]) is None


def test_k_nearest_neighbors():
    tree = PyRTree(dims=2)
    for point in [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]:
        tree.insert(point)
    k_nearest = tree.k_nearest_neighbors([0.1, 0.1], k=2)
    assert len(k_nearest) == 2
    assert np.allclose(k_nearest[0], [0.0, 0.0])
    assert np.allclose(k_nearest[1], [1.0, 1.0])


def test_neighbors_within_radius():
    tree = PyRTree(dims=2)
    for point in [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]:
        tree.insert(point)
    neighbors = tree.neighbors_within_radius([0.0, 0.0], radius=1.5)
    assert len(neighbors) == 2
    assert any(np.allclose(n, [0.0, 0.0]) for n in neighbors)
    assert any(np.allclose(n, [1.0, 0.0]) for n in neighbors)


def test_neighbors_within_radius_negative():
    with pytest.raises(ValueError):
        PyRTree(dims=2).neighbors_within_radius([0.0, 0.0], radius=-1.0)


def test_bulk_load_list():
    tree = PyRTree(dims=3)
    tree.bulk_load([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    assert tree.size() == 3


def test_bulk_load_numpy():
    tree = PyRTree(dims=2)
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    tree.bulk_load(pts)
    assert tree.size() == 3
    assert np.allclose(tree.nearest_neighbor([0.1, 0.1]), [0.0, 0.0])


def test_bulk_load_with_ids():
    tree = PyRTree(dims=2)
    tree.bulk_load([[0.0, 0.0], [5.0, 5.0]], data=[10, 20])
    dist, ids = tree.query(np.array([[0.0, 0.0]]), k=1)
    assert ids[0, 0] == 10


def test_bulk_load_id_length_mismatch():
    tree = PyRTree(dims=2)
    with pytest.raises(ValueError):
        tree.bulk_load([[0.0, 0.0], [1.0, 1.0]], data=[1])


def test_bulk_load_empty():
    tree = PyRTree(dims=2)
    tree.bulk_load([])
    assert tree.size() == 0


def test_locate_in_envelope():
    tree = PyRTree(dims=2)
    for point in [[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]]:
        tree.insert(point)
    found = tree.locate_in_envelope([-0.5, -0.5], [2.0, 2.0])
    assert len(found) == 2


def test_remove():
    tree = PyRTree(dims=2)
    tree.insert([1.0, 2.0])
    assert tree.size() == 1
    assert tree.remove([1.0, 2.0]) is True
    assert tree.size() == 0
    assert tree.remove([1.0, 2.0]) is False


def test_query_vectorized():
    tree = PyRTree(dims=2)
    tree.bulk_load([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]])
    pts = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
    dist, ids = tree.query(pts, k=2)
    assert dist.shape == (2, 2)
    assert ids.shape == (2, 2)
    # First query point's nearest is itself (id 0) at distance 0.
    assert ids[0, 0] == 0
    assert dist[0, 0] == pytest.approx(0.0)
    assert dist[0, 1] == pytest.approx(1.0)
    # Second query point's nearest is id 2.
    assert ids[1, 0] == 2


def test_query_fewer_than_k():
    tree = PyRTree(dims=2)
    tree.insert([0.0, 0.0])
    dist, ids = tree.query(np.array([[0.0, 0.0]]), k=3)
    assert ids[0, 0] == 0
    # Missing neighbours are padded.
    assert ids[0, 1] == -1
    assert ids[0, 2] == -1
    assert np.isinf(dist[0, 1])


def test_query_k_zero_raises():
    tree = PyRTree(dims=2)
    tree.insert([0.0, 0.0])
    with pytest.raises(ValueError):
        tree.query(np.array([[0.0, 0.0]]), k=0)


def test_query_radius_vectorized():
    tree = PyRTree(dims=2)
    tree.bulk_load([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    pts = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    result = tree.query_radius(pts, radius=1.5)
    assert len(result) == 2
    assert sorted(result[0]) == [0, 1]
    assert result[1] == [2]
