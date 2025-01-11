# rstar-python

Python bindings for the [rstar](https://github.com/georust/rstar) R*-tree spatial index library.

## Installation

```bash
pip install rstar-python
```

## Usage

```python
from rstar_python import PyRTree

# Create a 3D R-tree
tree = PyRTree(dims=3)

# Insert points
tree.insert([1.0, 2.0, 3.0])
tree.insert([4.0, 5.0, 6.0])

# Bulk load points
points = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
]
tree.bulk_load(points)

# Find nearest neighbor
nearest = tree.nearest_neighbor([1.1, 2.1, 3.1])

# Find k nearest neighbors
k_nearest = tree.k_nearest_neighbors([1.1, 2.1, 3.1], k=2)

# Find neighbors within radius
neighbors = tree.neighbors_within_radius([1.0, 2.0, 3.0], radius=1.0)

# Query points within a bounding box
points = tree.locate_in_envelope(
    min_corner=[0.0, 0.0, 0.0],
    max_corner=[2.0, 2.0, 2.0]
)

# Get tree size
size = tree.size()

# Remove a point
tree.remove([1.0, 2.0, 3.0])
```

## Features

- Supports 1D to 4D points
- Fast nearest neighbor queries
- Radius search
- Bounding box queries
- Bulk loading for faster initialization
- Built on top of the fast Rust rstar library

## Development

Requirements:
- Rust
- Python 3.7+
- maturin

```bash
# Clone repository
git clone https://github.com/kephale/rstar-python
cd rstar-python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install maturin pytest

# Build and install in development mode
maturin develop

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.