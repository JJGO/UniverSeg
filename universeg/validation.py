"""
Module containing utility functions for validating arguments using Pydantic.

Functions:
    - as_2tuple(val: size2t) -> Tuple[int, int]: Convert integer or 2-tuple to 2-tuple format.
    - validate_arguments_init(class_) -> class_: Decorator to validate the arguments of the __init__ method using Pydantic.
"""

from typing import Any, Dict, Tuple, Union

from pydantic import validate_arguments

size2t = Union[int, Tuple[int, int]]
Kwargs = Dict[str, Any]


def as_2tuple(val: size2t) -> Tuple[int, int]:
    """
    Convert integer or 2-tuple to 2-tuple format.

    Args:
        val (Union[int, Tuple[int, int]]): The value to convert.

    Returns:
        Tuple[int, int]: The converted 2-tuple.

    Raises:
        AssertionError: If val is not an integer or a 2-tuple with length 2.
    """
    if isinstance(val, int):
        return (val, val)
    assert isinstance(val, (list, tuple)) and len(val) == 2
    return tuple(val)


def validate_arguments_init(class_):
    """
    Decorator to validate the arguments of the __init__ method using Pydantic.

    Args:
        class_ (Any): The class to decorate.

    Returns:
        class_: The decorated class with validated __init__ method.
    """
    class_.__init__ = validate_arguments(class_.__init__)
    return class_
