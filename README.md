# HealPy Nest2Grid

A TensorFlow-based module for efficient transformation of HEALPix NESTED ordering data into structured grid tensor representations. This module provides utilities to convert HEALPix data into tensor formats suitable for deep learning and scientific computing applications.

## Features

- Transform HEALPix NESTED data into structured grid tensor representations
- Support for both single and multiple dataset transformations
- GPU-accelerated operations through TensorFlow
- Preserve HEALPix NESTED ordering in transformations
- Generate correct sky patch configurations in output tensors

## Installation

### Requirements
- TensorFlow >= 2.0
- Python >= 3.6

```bash
# Clone the repository
git clone https://github.com/yourusername/healpy_nest2grid.git
cd healpy_nest2grid

# Install dependencies
pip install tensorflow>=2.0
```

## Usage

### Basic Example

```python
import tensorflow as tf
from healpy_nest2grid import nest2grid_3d, grid2nest_1d_3d

# For single dataset transformation
# Input: HEALPix map in NESTED ordering with shape [12 * 4^k]
healpix_map = tf.random.normal([12 * 4**3])  # k=3 example
k = 3

# Transform to 3D tensor [12, 2^k, 2^k]
tensor_3d = nest2grid_3d(healpix_map, k)
print(tensor_3d.shape)  # [12, 8, 8]

# Transform back to original HEALPix format
healpix_map_restored = grid2nest_1d_3d(tensor_3d, k)
```

### Multiple Dataset Example

```python
# For multiple datasets
# Input: Multiple HEALPix maps [N, 12 * 4^k]
batch_size = 32
healpix_maps = tf.random.normal([batch_size, 12 * 4**3])
k = 3

# Transform to 4D tensor [N, 12, 2^k, 2^k]
tensor_4d = nest2grid_4d(healpix_maps, k)
print(tensor_4d.shape)  # [32, 12, 8, 8]

# Transform back to original format
healpix_maps_restored = grid2nest_2d(tensor_4d, k)
```

## API Reference

### Main Functions

- `nest2grid_3d(tensor_1d: TensorType, k: int) -> tf.Tensor`
  - Transforms 1D HEALPix NESTED data into 3D tensor [12, 2^k, 2^k]
  
- `grid2nest_1d_3d(tensor_3d: TensorType, k: int) -> tf.Tensor`
  - Converts 3D tensor back to 1D HEALPix NESTED format
  
- `nest2grid_4d(tensor_2d: TensorType, k: int) -> tf.Tensor`
  - Transforms 2D HEALPix NESTED data into 4D tensor [N, 12, 2^k, 2^k]
  
- `grid2nest_2d(tensor_4d: TensorType, k: int) -> tf.Tensor`
  - Converts 4D tensor back to 2D HEALPix NESTED format

### Helper Functions

- `compute_index(k: int) -> tf.Tensor`
  - Computes grid indices for a 2^k x 2^k grid from NESTED ordering
  
- `nest2grid_2d(tensor: TensorType, k: int) -> tf.Tensor`
  - Basic 1D to 2D grid transformation utility
  
- `grid2nest_1d(tensor_2d: TensorType, k: int) -> tf.Tensor`
  - Basic 2D grid to 1D NESTED transformation utility

## Performance

The module leverages TensorFlow's capabilities for:
- GPU acceleration
- Parallel processing
- Vectorized operations
- Automatic differentiation
- Both eager execution and graph mode compatibility

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [HEALPix Documentation](https://healpix.sourceforge.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/) 