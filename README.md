# rstar-python

[![PyPI](https://img.shields.io/pypi/v/rstar-python.svg)](https://pypi.org/project/rstar-python/)
[![Python versions](https://img.shields.io/pypi/pyversions/rstar-python.svg)](https://pypi.org/project/rstar-python/)
[![CI](https://github.com/kephale/rstar-python/actions/workflows/ci.yml/badge.svg)](https://github.com/kephale/rstar-python/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Python bindings for the [rstar](https://github.com/georust/rstar) R*-tree spatial index library.

A fast, **dynamic** spatial index for points and stored axis-aligned bounding
boxes in 2–8 dimensions. Insert and remove items after construction, run
nearest-neighbour and radius searches, and query spatial regions. Each item can
carry an integer id so queries can return references to *your* data.

## Installation

```bash
pip install rstar-python
```

Prebuilt `abi3` wheels are published for Linux, macOS, and Windows and work on
CPython 3.10+.

## Why rstar-python?

|                              | rstar-python | scipy `cKDTree` / sklearn `KDTree` | `Rtree` (libspatialindex) |
| ---------------------------- | :----------: | :--------------------------------: | :-----------------------: |
| Dynamic insert / remove      |      ✅       |        ❌ (static, rebuild)         |             ✅             |
| Bounding-box / region query  |      ✅       |          ❌ (radius only)           |             ✅             |
| Returns ids of matches       |      ✅       |          ✅ (row indices)           |             ✅             |
| Vectorised batch query       |      ✅       |                 ✅                  |             ❌             |
| Pure-Rust, no system C/C++ dep |    ✅       |       (C/Cython, bundled)          |   ❌ (needs libspatialindex) |

Reach for the KD-trees in scipy/scikit-learn when you build an index *once* from
a static array and only need point/radius queries. Reach for rstar-python when
you need to **mutate the index over time** or run **bounding-box queries** — with
easy prebuilt wheels and no C/C++ system dependency.

## Benchmarks

Measured with `python/benchmarks/benchmark_compare.py` (3-D, uniform random
data, best of 3) on an AMD Ryzen 7 PRO 5850U, Python 3.12, scipy 1.18 and
Rtree 1.4. Times are in milliseconds; lower is better.

| Workload | rstar-python | scipy `cKDTree` | `Rtree` |
| --- | ---: | ---: | ---: |
| Build from 200k points, 20k k-NN queries (k=8) | 125 | **100** | 2043 |
| Insert 20k points one by one, box query every 100 | **28** | 439 (rebuilt per query) | 614 |
| Bulk-load 200k boxes, 20k box intersection queries | **82** | — | 1422 |

scipy is faster for a static k-NN workload. rstar-python is faster when the
index changes over time or when you query boxes. `Rtree` has no batch query
API, so its timings include a Python loop over queries.

## Used by

- [royerlab/tracksdata](https://github.com/royerlab/tracksdata): spatial
  filtering of tracking graph nodes

## Usage

```python
import numpy as np
from rstar_python import BBoxRTree, RTree

# Create a 3D R-tree
tree = RTree(dims=3)

# Insert points. insert() returns the point's id.
tree.insert([1.0, 2.0, 3.0])            # -> 0  (auto-assigned)
tree.insert([4.0, 5.0, 6.0], data=42)   # -> 42 (explicit id)

# Or bulk-load (replaces existing contents). Accepts lists or numpy arrays,
# and an optional list of ids.
points = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]], dtype=np.float64)
tree.bulk_load(points, data=[10, 20, 30])

# --- Coordinate-returning queries ---
tree.nearest_neighbor([1.1, 2.1, 3.1])           # -> [1.0, 2.0, 3.0]
tree.k_nearest_neighbors([1.1, 2.1, 3.1], k=2)   # -> [[...], [...]]
tree.neighbors_within_radius([1.0, 2.0, 3.0], radius=1.0)
tree.locate_in_envelope(min_corner=[0, 0, 0], max_corner=[2, 2, 2])
tree.locate_in_envelope_ids([0, 0, 0], [2, 2, 2])  # stored ids, closed AABB

# Remove an exact point+id when coordinates are shared by multiple items.
tree.remove_item([1.0, 2.0, 3.0], data=10)

# --- Vectorised, id-returning queries (scipy/sklearn style) ---
query_pts = np.array([[1.1, 2.1, 3.1], [7.0, 8.0, 9.0]], dtype=np.float64)
distances, ids = tree.query(query_pts, k=2)
# distances: (2, 2) float64 Euclidean distances
# ids:       (2, 2) int64 ids; padded with -1 / inf if fewer than k exist

within = tree.query_radius(query_pts, radius=1.0)  # list of id lists

# --- Bookkeeping ---
len(tree)                  # number of points
tree.dims                  # 3
[1.0, 2.0, 3.0] in tree    # membership test
tree.remove([1.0, 2.0, 3.0])

# --- Stored bounding boxes ---
boxes = BBoxRTree(dims=2)
boxes.insert([0.0, 0.0], [2.0, 2.0], data=100)
boxes.insert([2.0, 1.0], [3.0, 3.0])
# Returns both ids: intersection uses closed bounds, so touching counts.
boxes.intersection([1.0, 1.0], [2.0, 2.0])

# Query many boxes in one native call. Results use CSR-style row offsets:
# the matches for row i are ids[offsets[i]:offsets[i + 1]].
query_mins = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64)
query_maxs = np.array([[2.0, 2.0], [6.0, 6.0]], dtype=np.float64)
ids, offsets = boxes.intersection_batch(query_mins, query_maxs)

boxes.remove_item([0.0, 0.0], [2.0, 2.0], data=100)

# Bulk loading replaces the stored boxes.
boxes.bulk_load(
    min_corners=[[0.0, 0.0], [5.0, 5.0]],
    max_corners=[[1.0, 1.0], [6.0, 6.0]],
    data=[10, 20],
)
```

`RTree` and `BBoxRTree` are the same classes as `PyRTree` and `PyBBoxRTree`;
both names work.

## Features

- Points and stored axis-aligned bounding boxes in 2–8 dimensions
- Dynamic insertion and exact item removal
- Per-item integer ids (auto-assigned or supplied)
- Nearest-neighbour and k-nearest-neighbour queries
- Radius search and axis-aligned bounding-box (envelope) queries
- Vectorised point `query` / `query_radius` and bounding-box
  `intersection_batch` queries over NumPy arrays
- NumPy-native batched intersection results without Python integer boxing
- Bulk loading (lists or NumPy arrays) for fast construction
- Type stubs (`py.typed`) for IDE and `mypy` support
- Built on the fast Rust [rstar](https://github.com/georust/rstar) library

## Development

Requirements:
- Rust (stable)
- Python 3.10+
- [maturin](https://github.com/PyO3/maturin)

```bash
git clone https://github.com/kephale/rstar-python
cd rstar-python

python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

pip install maturin pytest numpy

# Build and install in development mode
maturin develop --release

# Run tests
pytest python/tests -v

# Run the bounding-box performance benchmark
python python/benchmarks/benchmark_bbox_rtree.py

# Compare with scipy and Rtree (pip install scipy rtree)
python python/benchmarks/benchmark_compare.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
