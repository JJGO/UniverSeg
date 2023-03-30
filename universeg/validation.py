from typing import Any, Dict, Tuple, Union

from pydantic import validate_arguments

size2t = Union[int, Tuple[int, int]]
Kwargs = Dict[str, Any]


def as_2tuple(val):
    if isinstance(val, int):
        return (val, val)
    assert isinstance(val, (list, tuple)) and len(val) == 2
    return tuple(val)


# This decorator is necessary because decorating directly
# results in the class not being a valid type
def validate_arguments_init(class_):
    class_.__init__ = validate_arguments(class_.__init__)
    return class_
