"""Compare rstar-python with scipy cKDTree and Rtree (libspatialindex).

Three workloads:

1. static k-NN: build once from an array, then run a batch of k-NN queries.
2. dynamic: insert points one at a time, querying a box after every
   ``--query-every`` inserts. cKDTree cannot be updated, so it is rebuilt
   before each query.
3. boxes: bulk-load axis-aligned boxes, then run a batch of box
   intersection queries.

scipy and Rtree are optional; missing libraries are skipped.

    python python/benchmarks/benchmark_compare.py
"""

from __future__ import annotations

import argparse
import platform
import time

import numpy as np

from rstar_python import BBoxRTree, RTree

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None

try:
    from rtree import index as rtree_index
except ImportError:  # pragma: no cover
    rtree_index = None


def best_of(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def rtree_props(dims: int):
    props = rtree_index.Property()
    props.dimension = dims
    return props


def static_knn(rng, n: int, nq: int, k: int, dims: int, repeats: int) -> dict:
    points = rng.random((n, dims))
    queries = rng.random((nq, dims))
    out = {}

    def rstar():
        tree = RTree(dims=dims)
        tree.bulk_load(points)
        tree.query(queries, k=k)

    out["rstar-python"] = best_of(rstar, repeats)

    if cKDTree is not None:

        def kd():
            cKDTree(points).query(queries, k=k)

        out["scipy cKDTree"] = best_of(kd, repeats)

    if rtree_index is not None:

        def rt():
            stream = ((i, (*p, *p), None) for i, p in enumerate(points))
            idx = rtree_index.Index(stream, properties=rtree_props(dims))
            for q in queries:
                list(idx.nearest((*q, *q), k))

        out["Rtree"] = best_of(rt, repeats)
    return out


def dynamic(rng, n: int, query_every: int, dims: int, repeats: int) -> dict:
    points = rng.random((n, dims))
    lo = np.full(dims, 0.45)
    hi = np.full(dims, 0.55)
    out = {}

    def rstar():
        tree = RTree(dims=dims)
        for i, p in enumerate(points):
            tree.insert(p.tolist(), data=i)
            if i % query_every == 0:
                tree.locate_in_envelope_ids(lo.tolist(), hi.tolist())

    out["rstar-python"] = best_of(rstar, repeats)

    if cKDTree is not None:
        center = (lo + hi) / 2
        radius = float(np.linalg.norm(hi - center))

        def kd():
            for i in range(n):
                if i % query_every == 0:
                    # rebuild, then a radius query covering the box
                    cKDTree(points[: i + 1]).query_ball_point(center, radius)

        out["scipy cKDTree (rebuild)"] = best_of(kd, repeats)

    if rtree_index is not None:

        def rt():
            idx = rtree_index.Index(properties=rtree_props(dims))
            for i, p in enumerate(points):
                idx.insert(i, (*p, *p))
                if i % query_every == 0:
                    list(idx.intersection((*lo, *hi)))

        out["Rtree"] = best_of(rt, repeats)
    return out


def boxes(rng, n: int, nq: int, dims: int, repeats: int) -> dict:
    mins = rng.random((n, dims))
    maxs = mins + rng.random((n, dims)) * 0.01
    qmins = rng.random((nq, dims))
    qmaxs = qmins + 0.02
    out = {}

    def rstar():
        tree = BBoxRTree(dims=dims)
        tree.bulk_load(mins, maxs)
        tree.intersection_batch(qmins, qmaxs)

    out["rstar-python"] = best_of(rstar, repeats)

    if rtree_index is not None:

        def rt():
            stream = ((i, (*a, *b), None) for i, (a, b) in enumerate(zip(mins, maxs)))
            idx = rtree_index.Index(stream, properties=rtree_props(dims))
            for a, b in zip(qmins, qmaxs):
                list(idx.intersection((*a, *b)))

        out["Rtree"] = best_of(rt, repeats)
    return out


def report(title: str, results: dict) -> None:
    print(f"\n{title}")
    base = results["rstar-python"]
    for name, seconds in results.items():
        print(f"  {name:<26} {seconds * 1000:9.1f} ms  ({seconds / base:5.2f}x rstar-python)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--points", type=int, default=200_000)
    parser.add_argument("--queries", type=int, default=20_000)
    parser.add_argument("--dynamic-points", type=int, default=20_000)
    parser.add_argument("--query-every", type=int, default=100)
    parser.add_argument("--dims", type=int, default=3)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Python {platform.python_version()} on {platform.machine()}, {args.dims}-D, best of {args.repeats}")
    report(
        f"static k-NN: build {args.points:,} points, {args.queries:,} queries, k={args.k}",
        static_knn(rng, args.points, args.queries, args.k, args.dims, args.repeats),
    )
    report(
        f"dynamic: insert {args.dynamic_points:,} points one at a time, box query every {args.query_every}",
        dynamic(rng, args.dynamic_points, args.query_every, args.dims, args.repeats),
    )
    report(
        f"boxes: bulk-load {args.points:,} boxes, {args.queries:,} intersection queries",
        boxes(rng, args.points, args.queries, args.dims, args.repeats),
    )


if __name__ == "__main__":
    main()
