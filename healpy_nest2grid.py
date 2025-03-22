# healpy_nest2grid.py
"""HealPy Nest2Grid Module.

This module provides utilities for transforming HEALPix data in NESTED ordering format into
interleaved tensor representations. It supports both single dataset (1D input) and multiple
dataset (2D input) transformations.

Key Features:
- Transforms 1D HEALPix NESTED data into 3D tensor [12, 2^k, 2^k]
- Transforms 2D HEALPix NESTED data into 4D tensor [N, 12, 2^k, 2^k]
- Preserves NESTED ordering in the transformation
- The resulting 2^k x 2^k grids in the last two dimensions represent correct sky patch configurations

Implementation Details:
- Built on TensorFlow for efficient GPU acceleration and parallel processing
- Uses TensorFlow's optimized tensor operations for fast data transformation
- Supports automatic differentiation through the transformation operations
- Compatible with TensorFlow's eager execution and graph mode
- Efficiently handles large-scale HEALPix datasets through vectorized operations

The module uses TensorFlow operations for efficient tensor manipulations and bit operations
to maintain the proper ordering of HEALPix pixels. The output tensors are structured such that:
- First dimension (12) represents the base HEALPix faces
- Last two dimensions (2^k x 2^k) represent the hierarchical pixel subdivision within each face
- For multiple datasets, the first dimension (N) represents the batch size

For more information about HEALPix and NESTED ordering, see:
https://healpix.sourceforge.io/
"""

import tensorflow as tf
from typing import Union, Tuple

TensorType = Union[tf.Tensor, tf.Variable]

def compute_index(k: int) -> tf.Tensor:
    """Compute grid indices for a 2^k x 2^k grid from NESTED ordering.
    
    Args:
        k: Integer power for grid size (grid will be 2^k x 2^k)
        
    Returns:
        tf.Tensor: A 2D tensor containing the grid indices
    """
    n = 2**k
    rows = tf.tile(tf.expand_dims(tf.range(n), 1), [1, n])
    cols = tf.tile(tf.expand_dims(tf.range(n), 0), [n, 1])
    index = tf.zeros((n, n), dtype=tf.int32)
    for j in range(k):
        index |= tf.bitwise.left_shift(
            tf.bitwise.bitwise_and(tf.bitwise.right_shift(rows, j), 1), 2 * j + 1)
        index |= tf.bitwise.left_shift(
            tf.bitwise.bitwise_and(tf.bitwise.right_shift(cols, j), 1), 2 * j)
    return index

def nest2grid_2d(tensor: TensorType, k: int) -> tf.Tensor:
    """Convert a 1D NESTED ordered tensor into a 2D grid.
    
    Args:
        tensor: Input 1D tensor in NESTED ordering
        k: Integer power for grid size (output will be 2^k x 2^k)
        
    Returns:
        tf.Tensor: Rearranged 2D tensor in grid format
    """
    n = 2**k
    index = compute_index(k)
    return tf.gather(tensor, index)

def grid2nest_1d(tensor_2d: TensorType, k: int) -> tf.Tensor:
    """Convert a 2D grid tensor back to 1D NESTED ordering.
    
    Args:
        tensor_2d: Input 2D tensor in grid format
        k: Integer power that was used for the grid size
        
    Returns:
        tf.Tensor: 1D tensor in NESTED ordering
    """
    m = tf.range(0, 4**k, dtype=tf.int32)
    row = tf.zeros_like(m)
    col = tf.zeros_like(m)
    for j in range(k):
        col |= tf.bitwise.left_shift(
            tf.bitwise.bitwise_and(tf.bitwise.right_shift(m, 2 * j), 1), j)
        row |= tf.bitwise.left_shift(
            tf.bitwise.bitwise_and(tf.bitwise.right_shift(m, 2 * j + 1), 1), j)
    indices = tf.stack([row, col], axis=1)
    return tf.gather_nd(tensor_2d, indices)

def nest2grid_3d(tensor_1d: TensorType, k: int) -> tf.Tensor:
    """Convert a 1D NESTED ordered tensor into a 3D grid with HEALPix faces.
    
    Args:
        tensor_1d: Input 1D tensor in NESTED ordering
        k: Integer power for grid size
        
    Returns:
        tf.Tensor: 3D tensor with shape [12, 2^k, 2^k] in grid format
    """
    n = 2**k
    index_2d = compute_index(k)
    channel_offsets = tf.range(12, dtype=tf.int32)[:, None, None] * (4**k)
    indices = channel_offsets + tf.cast(index_2d[None, :, :], tf.int32)
    flat_indices = tf.reshape(indices, [-1])
    gathered = tf.gather(tensor_1d, flat_indices)
    return tf.reshape(gathered, [12, n, n])

def grid2nest_1d_3d(tensor_3d: TensorType, k: int) -> tf.Tensor:
    """Convert a 3D grid tensor back to 1D NESTED ordering.
    
    Args:
        tensor_3d: Input 3D tensor in grid format
        k: Integer power that was used for the grid size
        
    Returns:
        tf.Tensor: 1D tensor in NESTED ordering
    """
    m = tf.range(0, 4**k, dtype=tf.int32)
    j = tf.zeros_like(m)
    i = tf.zeros_like(m)
    for b in range(k):
        j |= tf.bitwise.left_shift(
            tf.bitwise.bitwise_and(tf.bitwise.right_shift(m, 2 * b), 1), b)
        i |= tf.bitwise.left_shift(
            tf.bitwise.bitwise_and(tf.bitwise.right_shift(m, 2 * b + 1), 1), b)
    c = tf.range(12, dtype=tf.int32)[:, None]
    indices = tf.stack([
        tf.broadcast_to(c, [12, 4**k]),
        tf.broadcast_to(i[None, :], [12, 4**k]),
        tf.broadcast_to(j[None, :], [12, 4**k])
    ], axis=2)
    indices = tf.reshape(indices, [-1, 3])
    return tf.gather_nd(tensor_3d, indices)

def nest2grid_4d(tensor_2d: TensorType, k: int) -> tf.Tensor:
    """Convert a 2D NESTED ordered tensor into a 4D grid with batches and HEALPix faces.
    
    Args:
        tensor_2d: Input 2D tensor in NESTED ordering
        k: Integer power for grid size
        
    Returns:
        tf.Tensor: 4D tensor with shape [B, 12, 2^k, 2^k] in grid format
    """
    B = tf.shape(tensor_2d)[0]
    C = 12
    N = 2**k
    M = 4**k
    index_2d = compute_index(k)
    col_indices = (tf.range(C, dtype=tf.int32)[:, None, None] * M + 
                   index_2d[None, :, :])
    col_indices = tf.tile(col_indices[None, :, :, :], [B, 1, 1, 1])
    b_indices = tf.tile(tf.range(B, dtype=tf.int32)[:, None, None, None], 
                        [1, C, N, N])
    indices = tf.stack([b_indices, col_indices], axis=-1)
    indices = tf.reshape(indices, [-1, 2])
    gathered = tf.gather_nd(tensor_2d, indices)
    return tf.reshape(gathered, [B, C, N, N])

def grid2nest_2d(tensor_4d: TensorType, k: int) -> tf.Tensor:
    """Convert a 4D grid tensor back to 2D NESTED ordering.
    
    Args:
        tensor_4d: Input 4D tensor in grid format
        k: Integer power that was used for the grid size
        
    Returns:
        tf.Tensor: 2D tensor in NESTED ordering
    """
    B = tf.shape(tensor_4d)[0]
    C = 12
    N = 2**k
    M = 4**k
    p = tf.range(C * M, dtype=tf.int32)
    c = p // M
    m = p % M
    i = tf.zeros_like(m)
    j = tf.zeros_like(m)
    for b in range(k):
        j |= tf.bitwise.left_shift(
            tf.bitwise.bitwise_and(tf.bitwise.right_shift(m, 2 * b), 1), b)
        i |= tf.bitwise.left_shift(
            tf.bitwise.bitwise_and(tf.bitwise.right_shift(m, 2 * b + 1), 1), b)
    b_indices = tf.range(B, dtype=tf.int32)[:, None] * tf.ones([1, C * M], dtype=tf.int32)
    indices = tf.stack([
        b_indices,
        tf.broadcast_to(c[None, :], [B, C * M]),
        tf.broadcast_to(i[None, :], [B, C * M]),
        tf.broadcast_to(j[None, :], [B, C * M])
    ], axis=2)
    indices = tf.reshape(indices, [-1, 4])
    gathered = tf.gather_nd(tensor_4d, indices)
    return tf.reshape(gathered, [B, C * M])

# Explicitly declare public API
__all__ = [
    'compute_index',
    'nest2grid_2d',
    'grid2nest_1d',
    'nest2grid_3d',
    'grid2nest_1d_3d',
    'nest2grid_4d',
    'grid2nest_2d',
] 