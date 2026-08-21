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


def test_ids_saturate_instead_of_wrapping():
    max_id = 2**63 - 1

    inserted = PyBBoxRTree(dims=2)
    assert inserted.insert([0.0, 0.0], [1.0, 1.0], data=max_id) == max_id
    assert inserted.insert([2.0, 2.0], [3.0, 3.0]) == max_id

    bulk_loaded = PyBBoxRTree(dims=2)
    bulk_loaded.bulk_load(
        np.array([[0.0, 0.0]], dtype=np.float64),
        np.array([[1.0, 1.0]], dtype=np.float64),
        data=np.array([max_id], dtype=np.int64),
    )
    assert bulk_loaded.insert([2.0, 2.0], [3.0, 3.0]) == max_id


def test_bulk_load_numpy_4d_replaces_contents():
    tree = PyBBoxRTree(dims=4)
    tree.insert([0.0] * 4, [1.0] * 4, data=99)
    # Fortran order exercises the strided ndarray fast path.
    mins = np.asfortranarray([[0.0] * 4, [5.0] * 4], dtype=np.float64)
    maxs = np.asfortranarray([[2.0] * 4, [6.0] * 4], dtype=np.float64)
    tree.bulk_load(mins, maxs, data=[20, 30])

    assert len(tree) == 2
    assert tree.intersection([1.0] * 4, [1.0] * 4) == [20]
    assert tree.intersection([5.5] * 4, [5.5] * 4) == [30]
    assert tree.insert([10.0] * 4, [11.0] * 4) == 100


def test_numpy_and_batched_intersection_results():
    tree = PyBBoxRTree(dims=2)
    mins = np.array([[0.0, 0.0], [2.0, 2.0], [8.0, 8.0]], dtype=np.float64)
    maxs = np.array([[4.0, 4.0], [6.0, 6.0], [9.0, 9.0]], dtype=np.float64)
    strided_ids = np.array([10, -1, 20, -1, 30], dtype=np.int64)[::2]
    tree.bulk_load(mins, maxs, data=strided_ids)

    query_mins = np.asfortranarray([[3.0, 3.0], [6.1, 6.1]], dtype=np.float64)
    query_maxs = np.asfortranarray([[8.0, 8.0], [7.9, 7.9]], dtype=np.float64)
    ids, offsets = tree.intersection_batch(query_mins, query_maxs)
    assert ids.dtype == offsets.dtype == np.int64
    assert offsets.tolist() == [0, 3, 3]
    assert sorted(ids[offsets[0] : offsets[1]].tolist()) == [10, 20, 30]

    empty_ids, empty_offsets = tree.intersection_batch(
        np.empty((0, 2), dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
    )
    assert empty_ids.tolist() == []
    assert empty_offsets.tolist() == [0]


def test_numpy_inputs_reject_wrong_dtypes():
    tree = PyBBoxRTree(dims=2)
    float32_corners = np.zeros((2, 2), dtype=np.float32)
    float64_corners = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match=r"min_corners is a numpy array of dtype float32; expected float64",
    ):
        tree.bulk_load(float32_corners, float64_corners)
    with pytest.raises(
        ValueError,
        match=r"data is a numpy array of dtype int32; expected int64",
    ):
        tree.bulk_load(
            float64_corners,
            float64_corners,
            data=np.arange(2, dtype=np.int32),
        )
    with pytest.raises(
        ValueError,
        match=r"min_corners is a numpy array of dtype float32; expected float64",
    ):
        tree.intersection_batch(float32_corners, float64_corners)


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
    with pytest.raises(ValueError, match="same number of rows"):
        tree.intersection_batch(
            np.empty((1, 2), dtype=np.float64),
            np.empty((2, 2), dtype=np.float64),
        )
