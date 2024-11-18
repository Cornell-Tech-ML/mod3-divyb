"""Module for tensor operations including mapping, zipping, and reducing tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    """Protocol for a map function that applies a callable to a Tensor."""

    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    """Class containing higher-order tensor operations such as map, zip, and reduce."""

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Higher-order tensor reduce function.

        Applies a reduction function over a specified dimension of a tensor.

        Args:
        ----
        fn: The right-most variable.
        start: The derivative we want to propagate backward to the leaves.

        Returns:
        -------
            A callable that takes a Tensor and a dimension index, returning a reduced Tensor.

        """
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False
    cuda = False


class TensorBackend:
    """Class to construct a tensor backend using specified tensor operations."""

    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda

    # def relu(self, t: Tensor) -> Tensor:
    #     """Applies the ReLU activation function to the input tensor.

    #     Args:
    #     ----
    #         t: Input tensor.

    #     Returns:
    #     -------
    #         A tensor with ReLU applied element-wise.

    #     """
    #     return np.maximum(0, t.storage)  # Assuming storage is a numpy array

    # def sigmoid(self, t: Tensor) -> Tensor:
    #     """Applies the sigmoid activation function to the input tensor.

    #     Args:
    #     ----
    #         t: Input tensor.

    #     Returns:
    #     -------
    #         A tensor with sigmoid applied element-wise.

    #     """
    #     return 1 / (1 + np.exp(-t.storage))  # Example implementation


class SimpleOps(TensorOps):
    """Class implementing simple tensor operations."""

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce function. ::

        fn_reduce = reduce(fn)
        out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            start (float): starting value for reduction
            a (:class:TensorData): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
        -------
            :class:TensorData : new tensor

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
    ----
        fn: function from float-to-float to apply

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        if len(in_shape) > len(out_shape):
            raise ValueError("in_shape must be <= out_shape")

        out_size = int(np.prod(out_shape))

        out_index = np.zeros_like(out_shape)
        in_index = np.zeros_like(in_shape)

        for i in range(out_size):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)

            in_pos = index_to_position(in_index, in_strides)
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = fn(in_storage[in_pos])

    return _map

    # : Implement for Task 2.3.
    # raise NotImplementedError("Need to implement for Task 2.3")


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip.

    This function applies a binary operation (fn) to two input tensors (a and b)
    and stores the result in the output tensor. It handles broadcasting of shapes.

    Args:
    ----
        fn: A function that takes two floats and returns a float.

    Returns:
    -------
        A tensor zip function that performs the operation on the input tensors.

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
        out_size = len(out)
        in_index_a = np.zeros(len(a_shape), dtype=np.int32)
        in_index_b = np.zeros(len(b_shape), dtype=np.int32)
        out_index = np.zeros(len(out_shape), dtype=np.int32)

        for i in range(out_size):
            # Convert the linear index to multi-dimensional index for output
            to_index(i, out_shape, out_index)

            # Convert the output index to input indices based on broadcasting
            # Ensure broadcasting logic correctly handles different shapes
            broadcast_index(out_index, out_shape, a_shape, in_index_a)
            broadcast_index(out_index, out_shape, b_shape, in_index_b)

            # Calculate positions in storage arrays
            a_position = index_to_position(in_index_a, a_strides)
            b_position = index_to_position(in_index_b, b_strides)
            out_position = index_to_position(out_index, out_strides)

            # Apply the function and store the result
            out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
    ----
        fn: reduction function mapping two floats to float

    Returns:
    -------
        Tensor reduce function.

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
        out_size = int(np.prod(out_shape))
        out_index = np.zeros_like(out_shape)

        for i in range(out_size):
            to_index(i, out_shape, out_index)
            assert out_index[reduce_dim] == 0
            out_val = 0.0

            for j in range(a_shape[reduce_dim]):
                a_index = np.copy(out_index)
                a_index[reduce_dim] = j

                a_pos = index_to_position(a_index, a_strides)

                if j == 0:
                    out_val = a_storage[a_pos]

                else:
                    out_val = fn(out_val, a_storage[a_pos])

            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = out_val

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
