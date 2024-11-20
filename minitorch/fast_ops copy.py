from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """
  # @njit(parallel=True) # type: ignore
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        
        assert len(in_shape) < MAX_DIMS and len(out_shape) < MAX_DIMS


        # TODO: Implement for Task 3.1.
        #raise NotImplementedError("Need to implement for Task 3.1")
        #Raising error if in_shape is not smaller than out_shape
        # Numba does not support formatted strings or f-strings

        if len(in_shape) > len(out_shape):
            raise ValueError("in_shape should be less or equal to out_shape" + str(len(in_shape))+ ">" + str(len(out_shape)))

        #check for stride allignment and equal shapes(Skip explicit indexing)
        shapes_align_check = (
            len(out_shape) == len(in_shape) and (out_shape == in_shape).all()
        )

        strides_align_check = (
            len(out_strides) == len(in_strides) and (out_strides == in_strides).all()
        )

        if shapes_align_check and strides_align_check:
            for i in prange(out.size):
                if i < len(in_storage):
                    out[i] = fn(in_storage[i])
            return
        
                # // Shared data structures such as (out, in_storage, out_index, in_index) can cause data race (i.e. when multiple threads write to the same memory location) if not properly synchronized. So, we define them as local variables to avoid data race
        # * Initialize as numpy arrays
        out_index = np.zeros_like(out_shape, dtype=np.int32)
        in_index = np.zeros_like(in_shape, dtype=np.int32)

        for i in prange(out.size):
            # Broadcast index from out_shape to in_shape
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Get the position of the current index in the storage arrays
            in_pos = index_to_position(in_index, in_strides)
            out_pos = index_to_position(out_index, out_strides)

            # Apply the function to the input value and store the result in the output array
            out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        assert (
            len(a_shape) < MAX_DIMS
            and len(b_shape) < MAX_DIMS
            and len(out_shape) < MAX_DIMS
        )

        # * If stride-aligned (a_strides == b_strides == out_strides) and shape-aligned, avoid explicit indexing
        stride_align_check = (
            len(out_strides) == len(a_strides) == len(b_strides)
            and (out_strides == a_strides).all()
            and (out_strides == b_strides).all()
        )
        shape_align_check = (
            len(out_shape) == len(a_shape) == len(b_shape)
            and (out_shape == a_shape).all()
            and (out_shape == b_shape).all()
        )
        if stride_align_check and shape_align_check:
            for i in prange(out.size):
                out[i] = fn(a_storage[i], b_storage[i])
            return

        # * If not stride-aligned, index explicitly; run loop in parallel
        else:
            # Define local variables to avoid data race. See note in tensor_map
            # * Initialize as numpy arrays to avoid repeated allocation
            out_index = np.zeros_like(out_shape, dtype=np.int32)
            a_index = np.zeros_like(a_shape, dtype=np.int32)
            b_index = np.zeros_like(b_shape, dtype=np.int32)

            for i in prange(out.size):
                # Broadcast index from out_shape to a_shape and b_shape
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)

                # Get the position of the current index in the storage arrays
                a_pos: int = index_to_position(a_index, a_strides)
                b_pos: int = index_to_position(b_index, b_strides)
                out_pos: int = index_to_position(out_index, out_strides)

                # Apply the function to the input values and store the result in the output array
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        #raise NotImplementedError("Need to implement for Task 3.1")
        assert len(out_shape) < MAX_DIMS and len(a_shape) < MAX_DIMS

        for i in prange(out.size):
            out_index: Index = np.zeros_like(out_shape)
            a_index: Index = np.zeros_like(a_shape)

            to_index(i, out_shape, out_index)
            to_index(i, out_shape, a_index)

            a_index[reduce_dim] = 0
            reduce_res: np.float64 = a_storage[index_to_position(a_index, a_strides)]

            for j in range(1, a_shape[reduce_dim]):
                a_index[reduce_dim] = j
                reduce_res = fn(
                    reduce_res, a_storage[index_to_position(a_index, a_strides)]
                )

            out[index_to_position(out_index, out_strides)] = reduce_res

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    for n in prange(out_shape[0]):  # Iterate over batch dimension
        for i in range(out_shape[1]):  # Iterate over rows of A
            for j in range(out_shape[2]):  # Iterate over columns of B
                sum = 0.0
                for k in range(a_shape[-1]):  # Iterate over shared dimension
                    # Ensure we are accessing the correct indices
                    a_index = (n, i, k)
                    b_index = (0, k, j)  # Adjusted to use the first (and only) batch for b

                    # Get positions in storage
                    a_pos = index_to_position(a_index, a_strides)
                    b_pos = index_to_position(b_index, b_strides)

                    # Accumulate the product
                    sum += a_storage[a_pos] * b_storage[b_pos]
                
                # Store the result in the output tensor
                out[index_to_position((n, i, j), out_strides)] = sum


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None