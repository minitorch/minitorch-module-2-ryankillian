"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """
    Multiply two numbers.

    This function takes two floating-point numbers and returns their product.
    It demonstrates a basic arithmetic operation implemented as a function.

    Args:
        x (float): The first number.
        y (float): The second number.

    Returns:
        float: The product of x and y.
    """
    return x * y


def id(x: float) -> float:
    """
    Return the input without any modification.

    This identity function demonstrates the simplest form of a function
    that returns exactly what it receives. It's often used for demonstrations
    and as a placeholder in computational graphs or during testing.

    Args:
        x (float): A numeric value.

    Returns:
        float: The same numeric value passed as input.
    """
    return x


def add(x: float, y: float) -> float:
    """
    Add two numbers.

    Args:
        x (float): The first number.
        y (float): The second number.

    Returns:
        float: The sum of x and y.
    """
    return x + y


def neg(x: float) -> float:
    """
    Negate the input value.

    Args:
        x (float): The number to negate.

    Returns:
        float: The negated value of x.
    """
    return float(-x)


def lt(x: float, y: float) -> float:
    """
    Determine if one number is less than another.

    It's used to perform element-wise comparisons in vectorized operations.

    Args:
        x (float): The first number to compare.
        y (float): The second number to compare against.

    Returns:
        float: 1.0 if x is less than y, otherwise 0.0.
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """
    Determine if two numbers are equal.

    It is useful for element-wise equality checks in numerical computations.

    Args:
        x (float): The first number to compare.
        y (float): The second number to compare.

    Returns:
        float: 1.0 if x is equal to y, otherwise 0.0.
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """
    Return the maximum of two numbers.

    Args:
        x (float): The first number to compare.
        y (float): The second number to compare.

    Returns:
        float: The greater number between x and y.
    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """
    Determine if two numbers are approximately equal to each other within a small tolerance.

    Args:
        x (float): The first number.
        y (float): The second number.

    Returns:
        bool: True if the difference between x and y is less than 0.01, else False.
    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """
    Compute the sigmoid activation function for a given input.

    The sigmoid function is defined as f(x) = 1 / (1 + exp(-x)). It maps any real-valued number
    to the (0, 1) interval, making it useful for tasks like binary classification. For stability
    in computation, especially to avoid overflow in exponential calculations, this implementation
    uses an alternative form for negative inputs: f(x) = exp(x) / (1 + exp(x)).

    Args:
        x (float): The input value to the sigmoid function.

    Returns:
        float: The output of the sigmoid function, ranging from 0 to 1.

    See Also:
        For more information on the sigmoid function, visit:
        https://en.wikipedia.org/wiki/Sigmoid_function
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def relu(x: float) -> float:
    """
    Apply the rectified linear unit function.

    This function returns x if x is greater than zero, otherwise it returns zero.
    It is commonly used as an activation function in neural networks.

    Args:
        x (float): The input value.

    Returns:
        float: The output of the ReLU function.

    See Also:
        https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """
    Compute the natural logarithm of a number, adjusted by a small constant for stability.

    This function calculates the natural logarithm (base e) of the input value. To enhance
    numerical stability and avoid a domain error (logarithm of zero), a small positive constant
    (EPS) is added to the input. This adjustment is crucial when dealing with very small or zero
    values which are common in various computational scenarios.

    Args:
        x (float): The input value for which the logarithm is to be calculated.

    Returns:
        float: The natural logarithm of `x + EPS`.
    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """
    Compute the gradient of the natural logarithm function for backpropagation.

    This function calculates the derivative of the natural logarithm function (log(x))
    with respect to x, which is 1/x. It then multiplies this derivative by the upstream
    gradient 'd', which is a common step in the backpropagation algorithm used in
    training neural networks.

    Args:
        x (float): The input value to the logarithm function. Must be positive to avoid a division by zero.
        d (float): The upstream gradient passed from the next layer or the loss gradient.

    Returns:
        float: The result of multiplying the derivative of log(x) by d, effectively d/x.

    Raises:
        ValueError: If x is less than or equal to zero, as log(x) is not defined for non-positive values.
    """
    if x <= 0:
        raise ValueError("log(x) is not defined for non-positive values.")
    return d / x


def inv(x: float) -> float:
    """
    Calculate the reciprocal of a number.

    Args:
        x (float): The number to find the reciprocal of.

    Returns:
        float: The reciprocal of x.

    Raises:
        ValueError: If x is zero, as division by zero is undefined.
    """
    if x == 0:
        raise ValueError("Division by zero is undefined.")
    return 1 / x


def inv_back(x: float, d: float) -> float:
    """
    Compute the gradient of the reciprocal function for backpropagation.

    Args:
        x (float): The input value to the reciprocal function. Must be non-zero to avoid division by zero.
        d (float): The upstream gradient passed from the next layer or the loss gradient.

    Returns:
        float: The result of multiplying the derivative of 1/x by d, effectively -d/x^2.

    Raises:
        ValueError: If x is zero, as division by zero results in an undefined derivative.
    """
    if x == 0:
        raise ValueError("Division by zero is undefined.")
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """
    Compute the gradient of the ReLU function for backpropagation.

    Args:
        x (float): The input value to the ReLU function.
        d (float): The upstream gradient.

    Returns:
        float: The gradient of ReLU with respect to input x, scaled by d.
    """
    return d * (1 if x > 0 else 0)


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map function that applies a given function `fn` to each element
    in an iterable of floats and returns a new list with the results.

    Args:
        fn (Callable[[float], float]): A function that takes a float and returns a float.

    Returns:
        Callable[[Iterable[float]], Iterable[float]]: A function that takes an iterable of floats,
        applies `fn` to each element, and returns a new list with the results.
    """

    def apply_map(values: Iterable[float]) -> Iterable[float]:
        return [fn(value) for value in values]

    return apply_map


def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Use `map` and `neg` to negate each element in `ls`.

    Args:
        ls (Iterable[float]): An iterable of floating-point numbers.

    Returns:
        Iterable[float]: An iterable of floating-point numbers where each element is the negation of the corresponding element in `ls`.
    """
    neg_map = map(neg)  # Get the mapping function for negation
    return neg_map(ls)  # Apply it to the list and return the new list


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Create a function that applies a binary function element-wise to two lists.

    Args:
        fn (Callable[[float, float], float]): A function that takes two floats and returns a float.

    Returns:
        Callable[[Iterable[float], Iterable[float]], Iterable[float]]: A function that takes two iterables of floats,
        applies `fn` to each pair of elements, and returns a new list containing the results of those applications.
    """

    def apply_zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        # Use the built-in zip function to pair elements from both lists and apply `fn` to each pair
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return apply_zipWith


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Add the elements of `ls1` and `ls2` using `zipWith` and `add`.

    Args:
        ls1 (Iterable[float]): First list of floats.
        ls2 (Iterable[float]): Second list of floats.

    Returns:
        Iterable[float]: A list containing the element-wise sum of `ls1` and `ls2`.
    """
    # You can directly return the result of invoking zipWith(add) on ls1 and ls2
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """
    Create a higher-order function for reduction. This function will apply a binary function
    recursively to combine all elements in a list starting from an initial value.

    Args:
        fn (Callable[[float, float], float]): A binary function that takes two floats and returns a float.
        start (float): The initial value to start the reduction.

    Returns:
        Callable[[Iterable[float]], float]: A function that when given a list of floats,
        reduces them into a single float using the binary function `fn` starting from `start`.
    """

    def apply_reduce(ls: Iterable[float]) -> float:
        result = start
        for value in ls:
            result = fn(result, value)
        return result

    return apply_reduce


def sum(ls: Iterable[float]) -> float:
    """
    Sum up a list using `reduce` and `add`.

    Args:
        ls (Iterable[float]): List of numbers to sum.

    Returns:
        float: The total sum of all numbers in the list.
    """
    # Create a reduce function for addition starting from 0
    sum_reduce = reduce(add, 0)
    return sum_reduce(ls)


def prod(ls: Iterable[float]) -> float:
    """
    Calculate the product of a list using `reduce` and `mul`.

    Args:
        ls (Iterable[float]): List of numbers to multiply.

    Returns:
        float: The product of all numbers in the list.
    """
    # Create a reduce function for multiplication, starting from 1 (the identity for multiplication)
    product_reduce = reduce(mul, 1)
    return product_reduce(ls)
