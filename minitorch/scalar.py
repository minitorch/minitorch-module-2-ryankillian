from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """
    `ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


class Scalar:
    """
    A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(
        self,
        v: float,
        back: ScalarHistory = ScalarHistory(),
        name: Optional[str] = None,
    ):
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        return "Scalar(%f)" % self.data

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> Scalar:
        return Add.apply(self, b)

    def __bool__(self) -> bool:
        return bool(self.data)

    def __lt__(self, b: ScalarLike) -> Scalar:
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:  # type: ignore[override]
        return EQ.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    def log(self) -> Scalar:
        return Log.apply(self)

    def exp(self) -> Scalar:
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        return ReLU.apply(self)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x: value to be accumulated
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        "True if this variable created by the user (no `last_fn`)"
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """
        Apply the chain rule to compute gradients for the inputs to this variable's
        function based on the output gradient.

        This method is crucial in the backward pass of automatic differentiation.
        It computes and accumulates gradients for each input variable involved in
        the last function that computed this variable's value.

        Args:
            d_output (Any): The derivative of the output with respect to this
                            variable. This is often the gradient flowing from the
                            output of the computation graph back to this variable.

        Returns:
            Iterable[Tuple[Variable, Any]]: A list of tuples where each tuple
                                            contains a variable and the gradient
                                            with respect to that variable. The list
                                            includes all variables that are not
                                            constants and thus need their gradients
                                            updated.

        Raises:
            AssertionError: If the variable does not have a history attribute set,
                            or if the last function or context is None, which would
                            imply that the variable was not created as a result of a
                            function that supports backpropagation.
        """
        # Retrieve the history of the operation that created this variable
        h = self.history
        assert h is not None, "History must not be None"
        assert h.last_fn is not None, "Last function must not be None"
        assert h.ctx is not None, "Context must not be None"

        # Use the backward method of the last function to get the derivatives
        # with respect to each of its inputs.
        local_derivatives = h.last_fn._backward(h.ctx, d_output)

        # Initialize a list to hold the results of the gradient computation.
        results = []

        # Loop through each input variable and its corresponding derivative
        for input_var, grad in zip(h.inputs, local_derivatives):
            # Ignore constants since they do not contribute to the gradients
            # in the context of learning parameters.
            if not input_var.is_constant():
                results.append((input_var, grad))

        return results

    def backward(self, d_output: Optional[float] = None) -> None:
        """
        Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """
    Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Parameters:
        f : function from n-scalars to 1-scalar.
        *scalars  : n input scalar values.
    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)  # type: ignore
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
