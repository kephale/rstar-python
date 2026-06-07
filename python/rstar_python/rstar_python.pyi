from typing import Sequence

import numpy as np
import numpy.typing as npt

__version__: str

class PyRTree:
    """An R*-tree spatial index over points of 2 to 8 dimensions.

    Each point carries an integer id (auto-assigned or supplied via ``data``).
    Coordinate-returning queries return coordinates; the vectorised
    ``query``/``query_radius`` return the stored ids.
    """

    def __init__(self, dims: int) -> None: ...
    @property
    def dims(self) -> int:
        """Number of dimensions this tree indexes."""

    def insert(self, point: Sequence[float], data: int | None = ...) -> int:
        """Insert a single point, returning its id.

        If ``data`` is omitted, an incrementing id is assigned automatically.
        """

    def bulk_load(
        self,
        points: Sequence[Sequence[float]] | npt.NDArray[np.float64],
        data: Sequence[int] | None = ...,
    ) -> None:
        """Build the tree from many points at once (replaces existing contents).

        ``data`` optionally supplies one id per point; otherwise ids are ``0..n``.
        """

    def nearest_neighbor(self, point: Sequence[float]) -> list[float] | None:
        """Coordinates of the single nearest point, or ``None`` if empty."""

    def k_nearest_neighbors(
        self, point: Sequence[float], k: int
    ) -> list[list[float]]:
        """Coordinates of the ``k`` nearest points, closest first."""

    def neighbors_within_radius(
        self, point: Sequence[float], radius: float
    ) -> list[list[float]]:
        """Coordinates of all points within ``radius`` (Euclidean)."""

    def locate_in_envelope(
        self, min_corner: Sequence[float], max_corner: Sequence[float]
    ) -> list[list[float]]:
        """Coordinates of all points inside the axis-aligned box."""

    def query(
        self, points: npt.NDArray[np.float64], k: int = ...
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Vectorised k-NN query (scipy/sklearn style).

        ``points`` is an ``(M, dims)`` array. Returns ``(distances, ids)`` of
        ``(M, k)`` arrays. Slots beyond the available points are ``inf`` / ``-1``.
        """

    def query_radius(
        self, points: npt.NDArray[np.float64], radius: float
    ) -> list[list[int]]:
        """Vectorised radius query: per query point, the ids within ``radius``."""

    def remove(self, point: Sequence[float]) -> bool:
        """Remove a point by coordinates. Returns ``True`` if one was removed."""

    def size(self) -> int:
        """Number of points in the tree."""

    def __len__(self) -> int: ...
    def __contains__(self, point: Sequence[float]) -> bool: ...
    def __repr__(self) -> str: ...
