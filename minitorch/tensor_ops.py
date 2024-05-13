from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor: ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:  # type: ignore
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:  # type: ignore
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:  # type: ignore
        pass

    @staticmethod  # type: ignore
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """

        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
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


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher-order tensor map function ::

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
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
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
        fn: Callable[[float, float], float]
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """
        Higher-order tensor zip function ::

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
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
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
        """
        Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
            :class:`TensorData` : new tensor
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
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map_simple(fn: Callable[[float], float]) -> Any:
    """
    Simple version of tensor map that applies a function `fn` element-wise
    to each element of the input tensor and stores the results in an output tensor.

    Args:
        fn (Callable[[float], float]): Unary function to apply to each element of the input tensor.

    Returns:
        Callable that processes tensor storage, applying fn element-wise.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Check that input and output shapes are identical
        if not np.array_equal(out_shape, in_shape):
            raise ValueError(
                "Input and output shapes must be identical for simple tensor_map."
            )

        # Iterate over each element in the tensor
        for i in range(np.prod(in_shape)):  # Loop over all elements (linearly)
            # Convert linear index to multi-dimensional index
            index = np.zeros(len(in_shape), dtype=int)
            to_index(i, in_shape, index)

            # Calculate linear storage position
            in_pos = index_to_position(index, in_strides)
            out_pos = index_to_position(index, out_strides)

            out[out_pos] = fn(in_storage[in_pos])

    return _map


def tensor_map(fn: Callable[[float], float]) -> Any:
    """
    Create a mapping operation to apply a function element-wise over a tensor.

    Args:
        fn (Callable[[float], float]): Function to apply to each element.

    Returns:
        Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]: A function that applies `fn`.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        for i in range(
            np.prod(out_shape)
        ):  # Iterate over every element in the output tensor
            out_index = np.zeros(len(out_shape), dtype=int)
            to_index(
                i, out_shape, out_index
            )  # Convert linear index to multi-dimensional index
            in_index = np.zeros(len(in_shape), dtype=int)
            broadcast_index(
                out_index, out_shape, in_shape, in_index
            )  # Handle broadcasting

            # Convert multi-dimensional index to linear index for both input and output
            in_pos = index_to_position(in_index, in_strides)
            out_pos = index_to_position(out_index, out_strides)

            # Apply the function and store the result
            out[out_pos] = fn(in_storage[in_pos])

    return _map


def tensor_zip_simple(fn: Callable[[float, float], float]) -> Any:
    """
    Simple version of tensor zip.

    Args:
        fn (Callable[[float, float], float]): Function to apply to elements of a and b.

    Returns:
        Callable that processes two tensors' storages, applying fn element-wise.
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
        # Iterate over each element in the tensor assuming a_shape and b_shape are the same as out_shape
        for i in range(np.prod(out_shape)):  # Loop over all elements (linearly)
            # Convert linear index to multi-dimensional index
            out_index = np.zeros(len(out_shape), dtype=int)
            to_index(i, out_shape, out_index)

            # Calculate linear storage positions for a and b
            a_pos = index_to_position(out_index, a_strides)
            b_pos = index_to_position(out_index, b_strides)

            # Apply the function fn to elements from a and b storages and store the result in out
            out[index_to_position(out_index, out_strides)] = fn(
                a_storage[a_pos], b_storage[b_pos]
            )

    return _zip


def tensor_zip(fn: Callable[[float, float], float]) -> Any:
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
        # Ensure the output shape is properly broadcasted
        broadcast_shape = shape_broadcast(a_shape, b_shape)

        # Loop through every element in the output tensor
        for i in range(np.prod(out_shape)):
            out_index = np.zeros(len(out_shape), dtype=int)
            to_index(
                i, out_shape, out_index
            )  # Convert linear index to multi-dimensional index

            # Calculate the broadcast indices for tensors a and b
            a_index = np.zeros(len(a_shape), dtype=int)
            b_index = np.zeros(len(b_shape), dtype=int)
            broadcast_index(out_index, broadcast_shape, a_shape, a_index)
            broadcast_index(out_index, broadcast_shape, b_shape, b_index)

            # Convert multi-dimensional index to linear index for a and b
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            # Apply the function and store the result in the output tensor
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip


# def tensor_reduce_simple(fn: Callable[[float, float], float]) -> Any:
#     def _reduce(
#         out: Storage,
#         out_shape: Shape,
#         out_strides: Strides,
#         a_storage: Storage,
#         a_shape: Shape,
#         a_strides: Strides,
#         reduce_dim: int,
#     ) -> None:
#         # Initialize the output storage with the identity or default value for the reduction operation
#         for i in range(np.prod(out_shape)):
#             index = np.zeros(len(out_shape), dtype=int)
#             to_index(i, out_shape, index)
#             pos = index_to_position(index, out_strides)
#             out[pos] = fn(
#                 out[pos], a_storage[pos]
#             )  # Assuming fn is correctly handling identity values

#         # Perform the reduction
#         dim_size = a_shape[reduce_dim]
#         for i in range(np.prod(a_shape)):
#             a_index = np.zeros(len(a_shape), dtype=int)
#             to_index(i, a_shape, a_index)
#             out_index = a_index.copy()
#             out_index[reduce_dim] = (
#                 0  # Reduce to the first element in the reduced dimension
#             )
#             a_pos = index_to_position(a_index, a_strides)
#             out_pos = index_to_position(out_index, out_strides)
#             out[out_pos] = fn(out[out_pos], a_storage[a_pos])

#     return _reduce


# def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
#     def _reduce(
#         out: Storage,
#         out_shape: Shape,
#         out_strides: Strides,
#         a_storage: Storage,
#         a_shape: Shape,
#         a_strides: Strides,
#         reduce_dim: int,
#     ) -> None:
#         # We will use a more complex strategy that can handle non-contiguous strides and potential parallelism
#         # Initialize the output storage
#         for i in range(np.prod(out_shape)):
#             out_index = np.zeros(len(out_shape), dtype=int)
#             to_index(i, out_shape, out_index)
#             out_pos = index_to_position(out_index, out_strides)
#             out[out_pos] = float('inf')  # or some appropriate identity value depending on fn

#         # Iterate over the input tensor
#         for i in range(np.prod(a_shape)):
#             a_index = np.zeros(len(a_shape), dtype=int)
#             to_index(i, a_shape, a_index)
#             out_index = a_index.copy()
#             out_index[reduce_dim] = 0
#             a_pos = index_to_position(a_index, a_strides)
#             out_pos = index_to_position(out_index, out_strides)
#             out[out_pos] = fn(out[out_pos], a_storage[a_pos])

#     return _reduce


def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
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
        assert len(out_shape) < MAX_DIMS and len(a_shape) < MAX_DIMS

        a_index: Index = np.array(a_shape)
        out_index: Index = np.array(out_shape)

        for i in range(len(out)):
            to_index(i, out_shape, out_index)  # Get out_index
            to_index(i, out_shape, a_index)  # Make a_index equals to out_index

            a_index[reduce_dim] = 0
            reduce_res: np.float64 = a_storage[index_to_position(a_index, a_strides)]
            for j in range(1, a_shape[reduce_dim]):

                a_index[reduce_dim] = j
                reduce_res = fn(
                    reduce_res, a_storage[index_to_position(a_index, a_strides)]
                )

            out[index_to_position(out_index, out_strides)] = reduce_res

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
