from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """

    # Calculate the position in the flattened array
    position = np.dot(index, strides)  # Dot product sums up the element-wise products

    return int(position)


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    This function calculates multi-dimensional indices based on the ordinal value and shape of the tensor.

    Args:
        ordinal (int): The ordinal position in the flattened tensor to convert.
        shape (Shape): The shape of the tensor.
        out_index (OutIndex): The output array where the computed index will be stored.
    """
    if ordinal < 0 or ordinal >= np.prod(shape):
        raise IndexingError("Ordinal out of bounds.")

    n = len(shape)
    for i in range(n):
        size = np.prod(shape[i + 1 :])  # Size of the dimension block
        out_index[i] = ordinal // size
        ordinal %= size


def broadcast_index(
    big_index: Index, big_shape: Shape, small_shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `small_shape` following broadcasting rules. This maps an index from a larger tensor
    to an index in a smaller tensor, handling dimensions that have been broadcasted.

    Args:
        big_index: Index in the larger tensor.
        big_shape: Shape of the larger tensor.
        small_shape: Shape of the smaller tensor to broadcast to.
        out_index: Output index in the smaller tensor shape.

    Returns:
        None: Fills `out_index` with the correct index values.
    """
    # Start by initializing out_index to zeros
    out_index.fill(0)

    # Calculate the difference in dimensions
    dim_diff = len(big_shape) - len(small_shape)

    # Iterate over each dimension in the smaller shape
    for i in range(len(small_shape)):
        if small_shape[i] == 1:
            # If the dimension is 1 in the smaller shape, it absorbs any index because of broadcasting
            out_index[i] = 0
        else:
            # Otherwise, directly map the index from the big tensor to the small tensor
            # making sure to account for the difference in the number of dimensions
            out_index[i] = big_index[i + dim_diff]

    # This function modifies out_index in place and does not return a value


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape that is compatible with both input shapes.
    Broadcasting is a way to apply operations to arrays with different shapes. The function
    aligns the shapes starting from the trailing dimensions, and dimensions are compatible when
    one of them is 1 or both are equal.

    Args:
        shape1 : first shape (tuple of int)
        shape2 : second shape (tuple of int)

    Returns:
        broadcasted shape (tuple of int)

    Raises:
        IndexingError : if the shapes cannot be broadcast according to the broadcasting rules
    """
    result = []
    for dim1, dim2 in zip(reversed(shape1), reversed(shape2)):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            raise IndexingError(f"Shapes {shape1} and {shape2} cannot be broadcast")
        result.append(max(dim1, dim2))

    # Handle remaining dimensions of the longer shape
    if len(shape1) > len(shape2):
        result.extend(reversed(shape1[: len(shape1) - len(shape2)]))
    else:
        result.extend(reversed(shape2[: len(shape2) - len(shape1)]))

    return tuple(reversed(result))


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor according to the specified order.

        Args:
            order (tuple of int): The desired order of the dimensions, where each element
                                specifies the new position of the dimension at that index.

        Returns:
            TensorData: A new TensorData instance with dimensions permuted according to 'order'.

        Raises:
            AssertionError: If the 'order' does not contain all dimensions exactly once.
        """
        # Ensure the order contains each dimension exactly once
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Order {order} does not match tensor dimensions {len(self.shape)}"

        # Calculate the new shape and strides
        new_shape = [self.shape[dim] for dim in order]
        new_strides = [self.strides[dim] for dim in order]

        # Create a new TensorData with the same storage but new shape and strides
        return TensorData(self._storage, tuple(new_shape), tuple(new_strides))

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
