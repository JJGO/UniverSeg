from typing import Callable

import einops as E
import torch
from torch import nn


def vmap(module: Callable, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Applies the given module over the initial batch dimension and the second (group) dimension.

    Args:
        module: a callable that is applied over the batch dimension and the second (group) dimension.
                must support batch operations
        x: tensor of shape (batch_size, group_size, ...).
        args: positional arguments to pass to `module`.
        kwargs: keyword arguments to pass to `module`.

    Returns:
        The output tensor with the same shape as the input tensor.
    """
    batch_size, group_size, *_ = x.shape
    grouped_input = E.rearrange(x, "B S ... -> (B S) ...")
    grouped_output = module(grouped_input, *args, **kwargs)
    output = E.rearrange(
        grouped_output, "(B S) ... -> B S ...", B=batch_size, S=group_size
    )
    return output


def vmap_fn(fn: Callable) -> Callable:
    """
    Returns a callable that applies the input function over the initial batch dimension and the second (group) dimension.

    Args:
        fn: function to apply over the batch dimension and the second (group) dimension.

    Returns:
        A callable that applies the input function over the initial batch dimension and the second (group) dimension.
    """

    def vmapped_fn(*args, **kwargs):
        return vmap(fn, *args, **kwargs)

    return vmapped_fn


class Vmap(nn.Module):
    def __init__(self, module: nn.Module):
        """
        Applies the given module over the initial batch dimension and the second (group) dimension.

        Args:
            module: module to apply over the batch dimension and the second (group) dimension.
        """
        super().__init__()
        self.vmapped = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the given module over the initial batch dimension and the second (group) dimension.

        Args:
            x: tensor of shape (batch_size, group_size, ...).

        Returns:
            The output tensor with the same shape as the input tensor.
        """
        return vmap(self.vmapped, x)


def vmap_cls(module_type: type) -> Callable:
    """
    Returns a callable that applies the input module type over the initial batch dimension and the second (group) dimension.

    Args:
        module_type: module type to apply over the batch dimension and the second (group) dimension.

    Returns:
        A callable that applies the input module type over the initial batch dimension and the second (group) dimension.
    """

    def vmapped_cls(*args, **kwargs):
        module = module_type(*args, **kwargs)
        return Vmap(module)

    return vmapped_cls
