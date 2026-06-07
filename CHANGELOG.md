# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-07

### Added
- Per-point integer ids: `insert(point, data=None)` returns the id (auto-assigned
  or explicit); `bulk_load(points, data=None)` accepts a list of ids.
- Vectorised `query(points, k=1)` returning `(distances, ids)` numpy arrays
  (scipy/sklearn `KDTree.query` style).
- Vectorised `query_radius(points, radius)` returning ids per query point.
- `bulk_load` now accepts a 2-D numpy `float64` array in addition to lists.
- Pythonic protocols: `__len__`, `__contains__`, `__repr__`, and a `dims` property.
- `__version__` attribute.
- Type stubs (`.pyi`) and `py.typed` marker for IDE / `mypy` support.
- `abi3` wheels (CPython 3.10+): one wheel per platform, forward-compatible with
  future Python versions.

### Changed
- Bumped `pyo3` 0.19 → 0.28 and `rstar` 0.12 → 0.13.
- Supported dimensions are now **2–8** (was 1–4). rstar 0.13 no longer supports
  1-dimensional trees; the upper bound was raised from 4 to 8.
- Fixed the Python package layout so wheels install and import correctly.

### Notes
- The coordinate-returning methods (`nearest_neighbor`, `k_nearest_neighbors`,
  `neighbors_within_radius`, `locate_in_envelope`) are unchanged and remain
  backward compatible.

## [0.0.1] - 2025-02-20

### Added
- Initial release of rstar-python
- Python bindings for the rstar R*-tree spatial index library
- Support for 2D and 3D spatial indexing
- Methods: insert, bulk_load, nearest, locate_within_distance, remove
- Support for Python 3.10, 3.11, and 3.12
- Wheels didn't work!