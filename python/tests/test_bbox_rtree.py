import numpy as np
import pytest

import rstar_python
from rstar_python import PyBBoxRTree


def test_public_export_and_protocols():
    assert rstar_python.PyBBoxRTree is PyBBoxRTree
    assert "PyBBoxRTree" in rstar_python.__all__

    tree = PyBBoxRTree(dims=2)
    assert tree.dims == 2
    assert tree.size() == len(tree) == 0
    assert repr(tree) == "PyBBoxRTree(dims=2, size=0)"


def test_insert_ids_and_intersection_semantics():
    tree = PyBBoxRTree(dims=2)
    assert tree.insert([0.0, 0.0], [4.0, 4.0]) == 0
    assert tree.insert([3.0, 3.0], [6.0, 6.0], data=10) == 10
    assert tree.insert([8.0, 8.0], [9.0, 9.0]) == 11

    # A query contained by the first box still intersects it; the second only touches.
    assert sorted(tree.intersection([1.0, 1.0], [3.0, 3.0])) == [0, 10]
    assert sorted(tree.intersection([6.0, 6.0], [8.0, 8.0])) == [10, 11]
    assert tree.intersection([6.1, 6.1], [7.9, 7.9]) == []


def test_bulk_load_numpy_4d_replaces_contents():
    tree = PyBBoxRTree(dims=4)
    tree.insert([0.0] * 4, [1.0] * 4, data=99)
    mins = np.array([[0.0] * 4, [5.0] * 4], dtype=np.float64)
    maxs = np.array([[2.0] * 4, [6.0] * 4], dtype=np.float64)
    tree.bulk_load(mins, maxs, data=[20, 30])

    assert len(tree) == 2
    assert tree.intersection([1.0] * 4, [1.0] * 4) == [20]
    assert tree.intersection([5.5] * 4, [5.5] * 4) == [30]
    assert tree.insert([10.0] * 4, [11.0] * 4) == 100


def test_remove_item_distinguishes_coincident_boxes():
    tree = PyBBoxRTree(dims=2)
    tree.insert([0.0, 0.0], [1.0, 1.0], data=10)
    tree.insert([0.0, 0.0], [1.0, 1.0], data=20)

    assert tree.remove_item([0.0, 0.0], [1.0, 1.0], 10) is True
    assert tree.intersection([0.5, 0.5], [0.5, 0.5]) == [20]
    assert tree.remove_item([0.0, 0.0], [1.0, 1.0], 10) is False


def test_empty_bulk_load_resets_contents():
    tree = PyBBoxRTree(dims=2)
    tree.insert([0.0, 0.0], [1.0, 1.0])
    tree.bulk_load([], [])
    assert len(tree) == 0
    assert tree.intersection([0.0, 0.0], [1.0, 1.0]) == []


@pytest.mark.parametrize("dims", [1, 9])
def test_invalid_tree_dimensions(dims):
    with pytest.raises(ValueError, match="Dimensions must be between 2 and 8"):
        PyBBoxRTree(dims=dims)


def test_dimension_row_and_id_mismatch_errors():
    tree = PyBBoxRTree(dims=2)
    with pytest.raises(ValueError, match="Expected 2 dimensions, got 3"):
        tree.insert([0.0, 0.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="same number of rows"):
        tree.bulk_load([[0.0, 0.0]], [])
    with pytest.raises(ValueError, match="Expected 1 ids.*got 2"):
        tree.bulk_load([[0.0, 0.0]], [[1.0, 1.0]], data=[1, 2])
    with pytest.raises(ValueError, match="max_corners row 0 must have 2 dimensions"):
        tree.bulk_load([[0.0, 0.0]], [[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="min_corners must have 2 columns, got 4"):
        tree.bulk_load(
            np.empty((0, 4), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )
