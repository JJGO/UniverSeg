from typing import Optional, Tuple, Union

import einops as E
import torch
import torch.nn as nn
from pydantic import validate_arguments

size2t = Union[int, Tuple[int, int]]


class CrossConv2d(nn.Conv2d):
    """
    Compute pairwise convolution between all element of x and all elements of y.
    x, y are tensors of size B,_,C,H,W where _ could be different number of elements in x and y
    essentially, we do a meshgrid of the elements to get B,Sx,Sy,C,H,W tensors, and then
    pairwise conv.
    Args:
        x (tensor): B,Sx,Cx,H,W
        y (tensor): B,Sy,Cy,H,W
    Returns:
        tensor: B,Sx,Sy,Cout,H,W
    """

    @validate_arguments
    def __init__(
        self,
        in_channels: size2t,
        out_channels: int,
        kernel_size: size2t,
        stride: size2t = 1,
        padding: size2t = 0,
        dilation: size2t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:

        if isinstance(in_channels, (list, tuple)):
            concat_channels = sum(in_channels)
        else:
            concat_channels = 2 * in_channels

        super().__init__(
            in_channels=concat_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, Sx, *_ = x.shape
        _, Sy, *_ = y.shape

        xs = E.repeat(x, "B Sx Cx H W -> B Sx Sy Cx H W", Sy=Sy)
        ys = E.repeat(y, "B Sy Cy H W -> B Sx Sy Cy H W", Sx=Sx)

        xy = torch.cat([xs, ys], dim=3,)

        batched_xy = E.rearrange(xy, "B Sx Sy C2 H W -> (B Sx Sy) C2 H W")
        batched_output = super().forward(batched_xy)

        output = E.rearrange(
            batched_output, "(B Sx Sy) Co H W -> B Sx Sy Co H W", B=B, Sx=Sx, Sy=Sy
        )
        return output
