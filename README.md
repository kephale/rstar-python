# rstar-python

Python bindings for the [rstar](https://github.com/georust/rstar) R*-tree spatial index library.

A fast, **dynamic** spatial index for points in 2–8 dimensions: insert and remove
points after construction, run nearest-neighbour and radius searches, and query
axis-aligned bounding boxes. Each point can carry an integer id so queries can
return references to *your* data, not just coordinates.

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

## Usage

```python
import numpy as np
from rstar_python import PyRTree

# Create a 3D R-tree
tree = PyRTree(dims=3)

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
```

## Features

- Points in 2–8 dimensions
- Dynamic `insert` / `remove`
- Per-point integer ids (auto-assigned or supplied)
- Nearest-neighbour and k-nearest-neighbour queries
- Radius search and axis-aligned bounding-box (envelope) queries
- Vectorised `query` / `query_radius` over a numpy array of points, returning
  distances and ids
- Bulk loading (lists or numpy arrays) for fast construction
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
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
