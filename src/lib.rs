use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rstar::{Point, RTree, RTreeObject, AABB};
use std::fmt::Debug;

// Type aliases for different dimensional points
type Point1D = [f64; 1];
type Point2D = [f64; 2];
type Point3D = [f64; 3];
type Point4D = [f64; 4];

// Generic wrapper around RTree
#[pyclass]
struct PyRTree {
    tree_1d: Option<RTree<Point1D>>,
    tree_2d: Option<RTree<Point2D>>,
    tree_3d: Option<RTree<Point3D>>,
    tree_4d: Option<RTree<Point4D>>,
    dims: usize,
}

#[pymethods]
impl PyRTree {
    #[new]
    fn new(dims: usize) -> PyResult<Self> {
        if dims < 1 || dims > 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Dimensions must be between 1 and 4",
            ));
        }
        
        let mut rtree = PyRTree {
            tree_1d: None,
            tree_2d: None,
            tree_3d: None,
            tree_4d: None,
            dims,
        };
        
        match dims {
            1 => rtree.tree_1d = Some(RTree::new()),
            2 => rtree.tree_2d = Some(RTree::new()),
            3 => rtree.tree_3d = Some(RTree::new()),
            4 => rtree.tree_4d = Some(RTree::new()),
            _ => unreachable!(),
        }
        
        Ok(rtree)
    }

    fn insert(&mut self, point: Vec<f64>) -> PyResult<()> {
        if point.len() != self.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} dimensions, got {}",
                self.dims,
                point.len()
            )));
        }

        match self.dims {
            1 => {
                if let Some(tree) = &mut self.tree_1d {
                    tree.insert([point[0]]);
                }
            }
            2 => {
                if let Some(tree) = &mut self.tree_2d {
                    tree.insert([point[0], point[1]]);
                }
            }
            3 => {
                if let Some(tree) = &mut self.tree_3d {
                    tree.insert([point[0], point[1], point[2]]);
                }
            }
            4 => {
                if let Some(tree) = &mut self.tree_4d {
                    tree.insert([point[0], point[1], point[2], point[3]]);
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn bulk_load(&mut self, points: Vec<Vec<f64>>) -> PyResult<()> {
        if points.is_empty() {
            return Ok(());
        }

        // Validate dimensions
        if points[0].len() != self.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} dimensions, got {}",
                self.dims,
                points[0].len()
            )));
        }

        match self.dims {
            1 => {
                let points_1d: Vec<Point1D> = points.into_iter().map(|p| [p[0]]).collect();
                self.tree_1d = Some(RTree::bulk_load(points_1d));
            }
            2 => {
                let points_2d: Vec<Point2D> = points.into_iter().map(|p| [p[0], p[1]]).collect();
                self.tree_2d = Some(RTree::bulk_load(points_2d));
            }
            3 => {
                let points_3d: Vec<Point3D> = points
                    .into_iter()
                    .map(|p| [p[0], p[1], p[2]])
                    .collect();
                self.tree_3d = Some(RTree::bulk_load(points_3d));
            }
            4 => {
                let points_4d: Vec<Point4D> = points
                    .into_iter()
                    .map(|p| [p[0], p[1], p[2], p[3]])
                    .collect();
                self.tree_4d = Some(RTree::bulk_load(points_4d));
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn nearest_neighbor(&self, point: Vec<f64>) -> PyResult<Option<Vec<f64>>> {
        if point.len() != self.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} dimensions, got {}",
                self.dims,
                point.len()
            )));
        }

        let result = match self.dims {
            1 => self
                .tree_1d
                .as_ref()
                .and_then(|tree| tree.nearest_neighbor(&[point[0]]))
                .map(|p| vec![p[0]]),
            2 => self
                .tree_2d
                .as_ref()
                .and_then(|tree| tree.nearest_neighbor(&[point[0], point[1]]))
                .map(|p| vec![p[0], p[1]]),
            3 => self
                .tree_3d
                .as_ref()
                .and_then(|tree| tree.nearest_neighbor(&[point[0], point[1], point[2]]))
                .map(|p| vec![p[0], p[1], p[2]]),
            4 => self
                .tree_4d
                .as_ref()
                .and_then(|tree| {
                    tree.nearest_neighbor(&[point[0], point[1], point[2], point[3]])
                })
                .map(|p| vec![p[0], p[1], p[2], p[3]]),
            _ => unreachable!(),
        };

        Ok(result)
    }

    fn k_nearest_neighbors(&self, point: Vec<f64>, k: usize) -> PyResult<Vec<Vec<f64>>> {
        if point.len() != self.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} dimensions, got {}",
                self.dims,
                point.len()
            )));
        }

        let mut result = match self.dims {
            1 => self
                .tree_1d
                .as_ref()
                .map(|tree| {
                    tree.nearest_neighbor_iter(&[point[0]])
                        .take(k)
                        .map(|p| vec![p[0]])
                        .collect()
                })
                .unwrap_or_default(),
            2 => self
                .tree_2d
                .as_ref()
                .map(|tree| {
                    tree.nearest_neighbor_iter(&[point[0], point[1]])
                        .take(k)
                        .map(|p| vec![p[0], p[1]])
                        .collect()
                })
                .unwrap_or_default(),
            3 => self
                .tree_3d
                .as_ref()
                .map(|tree| {
                    tree.nearest_neighbor_iter(&[point[0], point[1], point[2]])
                        .take(k)
                        .map(|p| vec![p[0], p[1], p[2]])
                        .collect()
                })
                .unwrap_or_default(),
            4 => self
                .tree_4d
                .as_ref()
                .map(|tree| {
                    tree.nearest_neighbor_iter(&[point[0], point[1], point[2], point[3]])
                        .take(k)
                        .map(|p| vec![p[0], p[1], p[2], p[3]])
                        .collect()
                })
                .unwrap_or_default(),
            _ => unreachable!(),
        };

        Ok(result)
    }

    fn neighbors_within_radius(&self, point: Vec<f64>, radius: f64) -> PyResult<Vec<Vec<f64>>> {
        if point.len() != self.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} dimensions, got {}",
                self.dims,
                point.len()
            )));
        }

        if radius < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Radius must be non-negative"
            ));
        }

        let squared_radius = radius * radius;
        let result = match self.dims {
            1 => self
                .tree_1d
                .as_ref()
                .map(|tree| {
                    tree.locate_within_distance([point[0]], squared_radius)
                        .map(|p| vec![p[0]])
                        .collect()
                })
                .unwrap_or_default(),
            2 => self
                .tree_2d
                .as_ref()
                .map(|tree| {
                    tree.locate_within_distance([point[0], point[1]], squared_radius)
                        .map(|p| vec![p[0], p[1]])
                        .collect()
                })
                .unwrap_or_default(),
            3 => self
                .tree_3d
                .as_ref()
                .map(|tree| {
                    tree.locate_within_distance([point[0], point[1], point[2]], squared_radius)
                        .map(|p| vec![p[0], p[1], p[2]])
                        .collect()
                })
                .unwrap_or_default(),
            4 => self
                .tree_4d
                .as_ref()
                .map(|tree| {
                    tree.locate_within_distance(
                        [point[0], point[1], point[2], point[3]],
                        squared_radius,
                    )
                    .map(|p| vec![p[0], p[1], p[2], p[3]])
                    .collect()
                })
                .unwrap_or_default(),
            _ => unreachable!(),
        };

        Ok(result)
    }

    fn size(&self) -> usize {
        match self.dims {
            1 => self.tree_1d.as_ref().map(|tree| tree.size()).unwrap_or(0),
            2 => self.tree_2d.as_ref().map(|tree| tree.size()).unwrap_or(0),
            3 => self.tree_3d.as_ref().map(|tree| tree.size()).unwrap_or(0),
            4 => self.tree_4d.as_ref().map(|tree| tree.size()).unwrap_or(0),
            _ => unreachable!(),
        }
    }

    fn remove(&mut self, point: Vec<f64>) -> PyResult<bool> {
        if point.len() != self.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} dimensions, got {}",
                self.dims,
                point.len()
            )));
        }

        let removed = match self.dims {
            1 => self
                .tree_1d
                .as_mut()
                .and_then(|tree| tree.remove(&[point[0]]))
                .is_some(),
            2 => self
                .tree_2d
                .as_mut()
                .and_then(|tree| tree.remove(&[point[0], point[1]]))
                .is_some(),
            3 => self
                .tree_3d
                .as_mut()
                .and_then(|tree| tree.remove(&[point[0], point[1], point[2]]))
                .is_some(),
            4 => self
                .tree_4d
                .as_mut()
                .and_then(|tree| tree.remove(&[point[0], point[1], point[2], point[3]]))
                .is_some(),
            _ => unreachable!(),
        };

        Ok(removed)
    }

    fn locate_in_envelope(&self, min_corner: Vec<f64>, max_corner: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        if min_corner.len() != self.dims || max_corner.len() != self.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} dimensions for corners, got {} and {}",
                self.dims,
                min_corner.len(),
                max_corner.len()
            )));
        }

        let result = match self.dims {
            1 => {
                let envelope = AABB::from_corners([min_corner[0]], [max_corner[0]]);
                self.tree_1d
                    .as_ref()
                    .map(|tree| {
                        tree.locate_in_envelope(&envelope)
                            .map(|p| vec![p[0]])
                            .collect()
                    })
                    .unwrap_or_default()
            }
            2 => {
                let envelope = AABB::from_corners(
                    [min_corner[0], min_corner[1]],
                    [max_corner[0], max_corner[1]],
                );
                self.tree_2d
                    .as_ref()
                    .map(|tree| {
                        tree.locate_in_envelope(&envelope)
                            .map(|p| vec![p[0], p[1]])
                            .collect()
                    })
                    .unwrap_or_default()
            }
            3 => {
                let envelope = AABB::from_corners(
                    [min_corner[0], min_corner[1], min_corner[2]],
                    [max_corner[0], max_corner[1], max_corner[2]],
                );
                self.tree_3d
                    .as_ref()
                    .map(|tree| {
                        tree.locate_in_envelope(&envelope)
                            .map(|p| vec![p[0], p[1], p[2]])
                            .collect()
                    })
                    .unwrap_or_default()
            }
            4 => {
                let envelope = AABB::from_corners(
                    [min_corner[0], min_corner[1], min_corner[2], min_corner[3]],
                    [max_corner[0], max_corner[1], max_corner[2], max_corner[3]],
                );
                self.tree_4d
                    .as_ref()
                    .map(|tree| {
                        tree.locate_in_envelope(&envelope)
                            .map(|p| vec![p[0], p[1], p[2], p[3]])
                            .collect()
                    })
                    .unwrap_or_default()
            }
            _ => unreachable!(),
        };

        Ok(result)
    }
}

#[pymodule]
fn rstar_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRTree>()?;
    Ok(())
}