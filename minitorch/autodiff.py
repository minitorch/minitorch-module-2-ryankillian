from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(
    f: Callable[..., float], *vals: float, arg: int = 0, epsilon: float = 1e-6
) -> float:
    """
    Computes an approximation to the derivative of `f` with respect to one arg.

    Args:
        f: Arbitrary function from n-scalar args to one value.
        *vals: n-float values (x_0, ..., x_{n-1}).
        arg: Index of the argument to compute the derivative with respect to.
        epsilon: A small constant for numerical stability.

    Returns:
        An approximation of the derivative of `f` with respect to the `arg`-th argument.
    """
    # Convert vals from tuple to list to allow modifications
    vals_list: List[float] = list(vals)

    # Increment the argument by epsilon for the forward step
    vals_plus: List[float] = vals_list[:]
    vals_plus[arg] += epsilon

    # Decrement the argument by epsilon for the backward step
    vals_minus: List[float] = vals_list[:]
    vals_minus[arg] -= epsilon

    # Calculate the function values for forward and backward steps
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    # Compute the central difference
    derivative = (f_plus - f_minus) / (2 * epsilon)

    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph using a depth-first search (DFS) approach.
    This function ensures that each variable is processed only after all variables that depend on it have been processed.

    Args:
        variable: The right-most variable from which to start the sort, typically the final output variable of the graph.

    Returns:
        An iterable of non-constant Variables in topological order, starting from the given variable and moving backwards.
    """
    visited = set()  # Set to keep track of visited nodes
    stack = []  # Stack to hold the topologically sorted variables

    def dfs(v: Variable) -> None:
        """Helper function to perform DFS"""
        if v.unique_id in visited:
            return
        visited.add(v.unique_id)

        # Iterate over the parents of the current variable
        for parent in v.parents:
            if not parent.is_constant():  # Only consider non-constant variables
                dfs(parent)

        stack.append(v)  # Append the variable to the stack after processing its parents

    # Start DFS from the given variable
    dfs(variable)

    # Since we want the elements in topological order starting from the right,
    # we need to reverse the stack because the deepest dependent variables are at the top.
    return reversed(stack)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Executes the backpropagation algorithm starting from a given variable, using
    a specified derivative as the initial gradient. This function calculates and
    distributes gradients back to all antecedent nodes in the computation graph.

    Args:
        variable (Variable): The variable from which to start backpropagation,
                             typically the final output of a computational graph.
        deriv (Any): The gradient of the output with respect to an external influence,
                     usually the derivative of a loss function with respect to the output.

    No return value. The gradients are accumulated directly into the `derivative`
    attribute of each leaf variable through the `accumulate_derivative` method.

    Detail:
        This function first computes a topological sort of all the nodes reachable
        from the `variable`, ensuring that each node's derivatives are computed before
        any of its dependents. It then traverses these nodes in reverse topological order,
        applying the chain rule to propagate derivatives back through the graph.
    """
    # Perform topological sort to ensure we process nodes in the correct order
    topsorted = topological_sort(variable)

    # Dictionary to store derivatives accumulated at each node
    intermediate_derivs: Dict[int, float] = defaultdict(float)
    intermediate_derivs[variable.unique_id] += deriv

    # Traverse the nodes in topological order
    for node in topsorted:
        d_out = intermediate_derivs[node.unique_id]

        # If node is a leaf, accumulate its derivative directly
        if node.is_leaf():
            node.accumulate_derivative(d_out)
        else:
            # For non-leaf nodes, apply the chain rule to calculate the derivatives
            # of the node's inputs and propagate them
            node_derivs = node.chain_rule(d_out)
            for scalar_input, input_deriv in node_derivs:
                intermediate_derivs[scalar_input.unique_id] += input_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
