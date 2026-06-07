use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rstar::primitives::GeomWithData;
use rstar::{RTree, AABB};

/// Minimum and maximum number of dimensions supported. rstar requires at
/// least 2 dimensions.
const MIN_DIMS: usize = 2;
const MAX_DIMS: usize = 8;

/// A point with an associated integer id, as stored in the tree.
type Item<const N: usize> = GeomWithData<[f64; N], i64>;

/// Return type of [`PyRTree::query`]: `(distances, ids)` numpy arrays.
type QueryResult<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>);

/// Convert a slice into a fixed-size point, validating its length.
fn to_point<const N: usize>(v: &[f64]) -> PyResult<[f64; N]> {
    <[f64; N]>::try_from(v)
        .map_err(|_| PyValueError::new_err(format!("Expected {} dimensions, got {}", N, v.len())))
}

// --- Generic, dimension-agnostic implementations ------------------------------
// Each helper is monomorphised per dimension; the `dispatch!` macro selects the
// right one at runtime based on the tree's dimensionality.

fn g_insert<const N: usize>(t: &mut RTree<Item<N>>, p: &[f64], id: i64) -> PyResult<()> {
    t.insert(GeomWithData::new(to_point::<N>(p)?, id));
    Ok(())
}

fn g_bulk<const N: usize>(points: &[Vec<f64>], ids: &[i64]) -> PyResult<RTree<Item<N>>> {
    let mut items = Vec::with_capacity(points.len());
    for (p, &id) in points.iter().zip(ids) {
        items.push(GeomWithData::new(to_point::<N>(p)?, id));
    }
    Ok(RTree::bulk_load(items))
}

fn g_nearest<const N: usize>(t: &RTree<Item<N>>, p: &[f64]) -> PyResult<Option<Vec<f64>>> {
    let q = to_point::<N>(p)?;
    Ok(t.nearest_neighbor(q).map(|g| g.geom().to_vec()))
}

fn g_knn<const N: usize>(t: &RTree<Item<N>>, p: &[f64], k: usize) -> PyResult<Vec<Vec<f64>>> {
    let q = to_point::<N>(p)?;
    Ok(t.nearest_neighbor_iter(q)
        .take(k)
        .map(|g| g.geom().to_vec())
        .collect())
}

fn g_radius<const N: usize>(t: &RTree<Item<N>>, p: &[f64], r: f64) -> PyResult<Vec<Vec<f64>>> {
    let q = to_point::<N>(p)?;
    Ok(t.locate_within_distance(q, r * r)
        .map(|g| g.geom().to_vec())
        .collect())
}

fn g_envelope<const N: usize>(
    t: &RTree<Item<N>>,
    min: &[f64],
    max: &[f64],
) -> PyResult<Vec<Vec<f64>>> {
    let envelope = AABB::from_corners(to_point::<N>(min)?, to_point::<N>(max)?);
    Ok(t.locate_in_envelope(envelope)
        .map(|g| g.geom().to_vec())
        .collect())
}

fn g_remove<const N: usize>(t: &mut RTree<Item<N>>, p: &[f64]) -> PyResult<bool> {
    let q = to_point::<N>(p)?;
    Ok(t.remove_at_point(q).is_some())
}

fn g_contains<const N: usize>(t: &RTree<Item<N>>, p: &[f64]) -> PyResult<bool> {
    let q = to_point::<N>(p)?;
    Ok(t.locate_at_point(q).is_some())
}

/// Vectorised k-nearest-neighbour query over many points at once.
/// Returns `(distances, ids)` as `(M, k)` arrays. Missing neighbours (when the
/// tree has fewer than `k` points) are filled with `inf` distance and id `-1`.
fn g_query<const N: usize>(
    t: &RTree<Item<N>>,
    points: ArrayView2<f64>,
    k: usize,
) -> PyResult<(Array2<f64>, Array2<i64>)> {
    if points.ncols() != N {
        return Err(PyValueError::new_err(format!(
            "Expected {} dimensions, got {}",
            N,
            points.ncols()
        )));
    }
    let m = points.nrows();
    let mut dists = Array2::<f64>::from_elem((m, k), f64::INFINITY);
    let mut ids = Array2::<i64>::from_elem((m, k), -1);
    for i in 0..m {
        let mut q = [0.0f64; N];
        for j in 0..N {
            q[j] = points[[i, j]];
        }
        for (slot, (g, d2)) in t
            .nearest_neighbor_iter_with_distance_2(q)
            .take(k)
            .enumerate()
        {
            dists[[i, slot]] = d2.sqrt();
            ids[[i, slot]] = g.data;
        }
    }
    Ok((dists, ids))
}

/// Vectorised radius query. Returns, per query point, the ids within `radius`.
fn g_query_radius<const N: usize>(
    t: &RTree<Item<N>>,
    points: ArrayView2<f64>,
    radius: f64,
) -> PyResult<Vec<Vec<i64>>> {
    if points.ncols() != N {
        return Err(PyValueError::new_err(format!(
            "Expected {} dimensions, got {}",
            N,
            points.ncols()
        )));
    }
    let r2 = radius * radius;
    let m = points.nrows();
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let mut q = [0.0f64; N];
        for j in 0..N {
            q[j] = points[[i, j]];
        }
        out.push(t.locate_within_distance(q, r2).map(|g| g.data).collect());
    }
    Ok(out)
}

// --- The dimension-tagged tree storage ----------------------------------------

macro_rules! define_tree {
    ($(($variant:ident, $n:literal)),+ $(,)?) => {
        enum Tree { $($variant(RTree<Item<$n>>)),+ }

        impl Tree {
            fn empty(dims: usize) -> PyResult<Tree> {
                match dims {
                    $($n => Ok(Tree::$variant(RTree::new())),)+
                    _ => Err(PyValueError::new_err(format!(
                        "Dimensions must be between {} and {}, got {}", MIN_DIMS, MAX_DIMS, dims
                    ))),
                }
            }

            fn bulk(dims: usize, points: &[Vec<f64>], ids: &[i64]) -> PyResult<Tree> {
                match dims {
                    $($n => Ok(Tree::$variant(g_bulk::<$n>(points, ids)?)),)+
                    _ => Err(PyValueError::new_err(format!(
                        "Dimensions must be between {} and {}, got {}", MIN_DIMS, MAX_DIMS, dims
                    ))),
                }
            }

            fn dims(&self) -> usize {
                match self { $(Tree::$variant(_) => $n),+ }
            }
        }
    };
}

define_tree!(
    (D2, 2),
    (D3, 3),
    (D4, 4),
    (D5, 5),
    (D6, 6),
    (D7, 7),
    (D8, 8),
);

/// Run `$body` against the inner `RTree`, regardless of its dimension.
macro_rules! dispatch {
    ($e:expr, $t:ident => $body:expr) => {
        match $e {
            Tree::D2($t) => $body,
            Tree::D3($t) => $body,
            Tree::D4($t) => $body,
            Tree::D5($t) => $body,
            Tree::D6($t) => $body,
            Tree::D7($t) => $body,
            Tree::D8($t) => $body,
        }
    };
}

/// An R*-tree spatial index over points of 2 to 8 dimensions.
///
/// Each point carries an integer id (auto-assigned or supplied via ``data``).
/// Coordinate-returning queries (``nearest_neighbor`` etc.) return coordinates,
/// while the vectorised ``query``/``query_radius`` return the stored ids.
#[pyclass]
struct PyRTree {
    tree: Tree,
    next_id: i64,
}

#[pymethods]
impl PyRTree {
    #[new]
    fn new(dims: usize) -> PyResult<Self> {
        Ok(PyRTree {
            tree: Tree::empty(dims)?,
            next_id: 0,
        })
    }

    /// Insert a single point, returning its id.
    /// If ``data`` is omitted, an incrementing id is assigned automatically.
    #[pyo3(signature = (point, data = None))]
    fn insert(&mut self, point: Vec<f64>, data: Option<i64>) -> PyResult<i64> {
        let id = data.unwrap_or(self.next_id);
        dispatch!(&mut self.tree, t => g_insert(t, &point, id)?);
        self.next_id = self.next_id.max(id + 1);
        Ok(id)
    }

    /// Build the tree from many points at once (replaces existing contents).
    /// ``points`` may be a list of lists or a 2-D numpy ``float64`` array.
    /// ``data`` optionally supplies one id per point; otherwise ids are ``0..n``.
    #[pyo3(signature = (points, data = None))]
    fn bulk_load(&mut self, points: &Bound<'_, PyAny>, data: Option<Vec<i64>>) -> PyResult<()> {
        let pts = extract_points(points)?;
        if pts.is_empty() {
            self.tree = Tree::empty(self.tree.dims())?;
            return Ok(());
        }
        let ids: Vec<i64> = match data {
            Some(d) => {
                if d.len() != pts.len() {
                    return Err(PyValueError::new_err(format!(
                        "Expected {} ids to match {} points, got {}",
                        pts.len(),
                        pts.len(),
                        d.len()
                    )));
                }
                d
            }
            None => (0..pts.len() as i64).collect(),
        };
        let max_id = ids.iter().copied().max().unwrap_or(-1);
        self.tree = Tree::bulk(self.tree.dims(), &pts, &ids)?;
        self.next_id = self.next_id.max(max_id + 1);
        Ok(())
    }

    /// Return the coordinates of the single nearest point, or ``None`` if empty.
    fn nearest_neighbor(&self, point: Vec<f64>) -> PyResult<Option<Vec<f64>>> {
        dispatch!(&self.tree, t => g_nearest(t, &point))
    }

    /// Return the coordinates of the ``k`` nearest points, closest first.
    fn k_nearest_neighbors(&self, point: Vec<f64>, k: usize) -> PyResult<Vec<Vec<f64>>> {
        dispatch!(&self.tree, t => g_knn(t, &point, k))
    }

    /// Return the coordinates of all points within ``radius`` (Euclidean).
    fn neighbors_within_radius(&self, point: Vec<f64>, radius: f64) -> PyResult<Vec<Vec<f64>>> {
        if radius < 0.0 {
            return Err(PyValueError::new_err("Radius must be non-negative"));
        }
        dispatch!(&self.tree, t => g_radius(t, &point, radius))
    }

    /// Return the coordinates of all points inside the axis-aligned box.
    fn locate_in_envelope(
        &self,
        min_corner: Vec<f64>,
        max_corner: Vec<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        dispatch!(&self.tree, t => g_envelope(t, &min_corner, &max_corner))
    }

    /// Vectorised k-NN query (scipy/sklearn style).
    ///
    /// ``points`` is an ``(M, dims)`` numpy ``float64`` array. Returns a tuple
    /// ``(distances, ids)`` of ``(M, k)`` arrays: Euclidean distances and the
    /// stored ids. Slots beyond the available points are ``inf`` / ``-1``.
    #[pyo3(signature = (points, k = 1))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        k: usize,
    ) -> PyResult<QueryResult<'py>> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be >= 1"));
        }
        let view = points.as_array();
        let (dists, ids) = dispatch!(&self.tree, t => g_query(t, view, k)?);
        Ok((dists.into_pyarray(py), ids.into_pyarray(py)))
    }

    /// Vectorised radius query. ``points`` is an ``(M, dims)`` numpy array;
    /// returns a list of ``M`` lists, each holding the ids within ``radius``.
    fn query_radius(
        &self,
        points: PyReadonlyArray2<'_, f64>,
        radius: f64,
    ) -> PyResult<Vec<Vec<i64>>> {
        if radius < 0.0 {
            return Err(PyValueError::new_err("Radius must be non-negative"));
        }
        let view = points.as_array();
        dispatch!(&self.tree, t => g_query_radius(t, view, radius))
    }

    /// Remove a point by its coordinates. Returns ``True`` if one was removed.
    fn remove(&mut self, point: Vec<f64>) -> PyResult<bool> {
        dispatch!(&mut self.tree, t => g_remove(t, &point))
    }

    /// Number of dimensions this tree indexes.
    #[getter]
    fn dims(&self) -> usize {
        self.tree.dims()
    }

    /// Number of points in the tree.
    fn size(&self) -> usize {
        dispatch!(&self.tree, t => t.size())
    }

    fn __len__(&self) -> usize {
        self.size()
    }

    fn __contains__(&self, point: Vec<f64>) -> PyResult<bool> {
        dispatch!(&self.tree, t => g_contains(t, &point))
    }

    fn __repr__(&self) -> String {
        format!("PyRTree(dims={}, size={})", self.tree.dims(), self.size())
    }
}

/// Accept either a list-of-lists or a 2-D numpy ``float64`` array.
fn extract_points(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        Ok(arr
            .as_array()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect())
    } else {
        obj.extract::<Vec<Vec<f64>>>().map_err(|_| {
            PyValueError::new_err("points must be a list of lists or a 2-D float64 numpy array")
        })
    }
}

#[pymodule]
fn rstar_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRTree>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
