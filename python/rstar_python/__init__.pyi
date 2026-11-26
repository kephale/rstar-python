from .rstar_python import PyBBoxRTree as PyBBoxRTree
from .rstar_python import PyRTree as PyRTree
from .rstar_python import __version__ as __version__

RTree = PyRTree
BBoxRTree = PyBBoxRTree

__all__ = ["BBoxRTree", "PyBBoxRTree", "PyRTree", "RTree", "__version__"]
