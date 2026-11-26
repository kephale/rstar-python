from .rstar_python import PyBBoxRTree, PyRTree, __version__

# Shorter names. The Py-prefixed names remain supported.
RTree = PyRTree
BBoxRTree = PyBBoxRTree

__all__ = ["BBoxRTree", "PyBBoxRTree", "PyRTree", "RTree", "__version__"]
