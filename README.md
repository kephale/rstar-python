# rstar-python

```
from rstar_python import PyRTree

# Create a 3D tree
tree = PyRTree(dims=3)

# Insert points
tree.insert([1.0, 2.0, 3.0])
tree.insert([4.0, 5.0, 6.0])

# Find nearest neighbors
nearest = tree.nearest_neighbor([1.1, 2.1, 3.1])
k_nearest = tree.k_nearest_neighbors([1.1, 2.1, 3.1], k=2)
radius_neighbors = tree.neighbors_within_radius([1.0, 2.0, 3.0], radius=1.0)
```