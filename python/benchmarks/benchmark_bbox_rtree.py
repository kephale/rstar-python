"""Reproducible microbenchmark for bounding-box bulk loads and intersections."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from rstar_python import PyBBoxRTree

FloatArray = npt.NDArray[np.float64]


def make_boxes(
    rng: np.random.Generator, rows: int, dims: int, size: float
) -> tuple[FloatArray, FloatArray]:
    mins = rng.random((rows, dims), dtype=np.float64)
    maxs = mins + rng.random((rows, dims), dtype=np.float64) * size
    return mins, maxs


def median_seconds(function: Callable[[], Any], repeats: int) -> float:
    function()  # Warm native code and allocations before measuring.
    samples: list[float] = []
    for _ in range(repeats):
        gc.disable()
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
        gc.enable()
        del result
    return statistics.median(samples)


def run_benchmark(
    num_boxes: int, num_queries: int, dims: int, repeats: int
) -> dict[str, float | int]:
    rng = np.random.default_rng(42)
    mins, maxs = make_boxes(rng, num_boxes, dims, 0.005)
    query_mins, query_maxs = make_boxes(rng, num_queries, dims, 0.05)
    ids = np.arange(num_boxes, dtype=np.int64)

    def build() -> PyBBoxRTree:
        tree = PyBBoxRTree(dims)
        tree.bulk_load(mins, maxs, ids)
        return tree

    bulk_seconds = median_seconds(build, repeats)
    tree = build()

    def scalar_list() -> int:
        return sum(
            len(tree.intersection(lower, upper))
            for lower, upper in zip(query_mins, query_maxs, strict=True)
        )

    def scalar_numpy() -> int:
        return sum(
            len(tree.intersection_numpy(lower, upper))
            for lower, upper in zip(query_mins, query_maxs, strict=True)
        )

    def batch() -> int:
        result_ids, offsets = tree.intersection_batch(query_mins, query_maxs)
        assert offsets[-1] == len(result_ids)
        return len(result_ids)

    hits = scalar_list()
    assert scalar_numpy() == batch() == hits
    scalar_list_seconds = median_seconds(scalar_list, repeats)
    scalar_numpy_seconds = median_seconds(scalar_numpy, repeats)
    batch_seconds = median_seconds(batch, repeats)

    return {
        "boxes": num_boxes,
        "queries": num_queries,
        "dims": dims,
        "hits": hits,
        "bulk_seconds": bulk_seconds,
        "scalar_list_seconds": scalar_list_seconds,
        "scalar_numpy_seconds": scalar_numpy_seconds,
        "batch_seconds": batch_seconds,
        "batch_speedup_over_list": scalar_list_seconds / batch_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxes", type=int, default=200_000)
    parser.add_argument("--queries", type=int, default=20_000)
    parser.add_argument("--dims", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(
        json.dumps(
            run_benchmark(args.boxes, args.queries, args.dims, args.repeats),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
