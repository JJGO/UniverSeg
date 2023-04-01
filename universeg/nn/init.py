import warnings
from typing import Optional

import torch
from torch import nn
from torch.nn import init


def initialize_weight(
    weight: torch.Tensor,
    distribution: Optional[str],
    nonlinearity: Optional[str] = "LeakyReLU",
) -> None:
    """Initialize the weight tensor with a chosen distribution and nonlinearity.

    Args:
        weight (torch.Tensor): The weight tensor to initialize.
        distribution (Optional[str]): The distribution to use for initialization. Can be one of "zeros",
            "kaiming_normal", "kaiming_uniform", "kaiming_normal_fanout", "kaiming_uniform_fanout",
            "glorot_normal", "glorot_uniform", or "orthogonal".
        nonlinearity (Optional[str]): The type of nonlinearity to use. Can be one of "LeakyReLU", "Sine",
            "Tanh", "Silu", or "Gelu".

    Returns:
        None
    """

    if distribution is None:
        return

    if nonlinearity:
        nonlinearity = nonlinearity.lower()
        if nonlinearity == "leakyrelu":
            nonlinearity = "leaky_relu"

    if nonlinearity == "sine":
        warnings.warn("sine gain not implemented, defaulting to tanh")
        nonlinearity = "tanh"

    if nonlinearity is None:
        nonlinearity = "linear"

    if nonlinearity in ("silu", "gelu"):
        nonlinearity = "leaky_relu"

    gain = 1 if nonlinearity is None else init.calculate_gain(nonlinearity)

    if distribution == "zeros":
        init.zeros_(weight)
    elif distribution == "kaiming_normal":
        init.kaiming_normal_(weight, nonlinearity=nonlinearity)
    elif distribution == "kaiming_uniform":
        init.kaiming_uniform_(weight, nonlinearity=nonlinearity)
    elif distribution == "kaiming_normal_fanout":
        init.kaiming_normal_(weight, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "kaiming_uniform_fanout":
        init.kaiming_uniform_(weight, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "glorot_normal":
        init.xavier_normal_(weight, gain=gain)
    elif distribution == "glorot_uniform":
        init.xavier_uniform_(weight, gain)
    elif distribution == "orthogonal":
        init.orthogonal_(weight, gain)
    else:
        raise ValueError(f"Unsupported distribution '{distribution}'")


def initialize_bias(
    bias: torch.Tensor,
    distribution: Optional[float] = 0,
    nonlinearity: Optional[str] = "LeakyReLU",
    weight: Optional[torch.Tensor] = None,
) -> None:
    """Initialize the bias tensor with a constant or a chosen distribution and nonlinearity.

    Args:
        bias (torch.Tensor): The bias tensor to initialize.
        distribution (Optional[float]): The constant value to initialize the bias to.
        nonlinearity (Optional[str]): The type of nonlinearity to use when initializing the bias.
        weight (Optional[torch.Tensor]): The weight tensor to use when initializing the bias.

    Returns:
        None
    """

    if distribution is None:
        return

    if isinstance(distribution, (int, float)):
        init.constant_(bias, distribution)
    else:
        raise NotImplementedError(f"Unsupported distribution '{distribution}'")


def initialize_layer(
    layer: nn.Module,
    distribution: Optional[str] = "kaiming_normal",
    init_bias: Optional[float] = 0,
    nonlinearity: Optional[str] = "LeakyReLU",
) -> None:
    """Initialize the weight and bias tensors of a linear or convolutional layer.

    Args:
        layer (nn.Module): The layer to initialize.
        distribution (Optional[str]): The distribution to use for weight initialization.
        init_bias (Optional[float]): The value to use for bias initialization.
        nonlinearity (Optional[str]): The type of nonlinearity to use when initializing the layer.

    Returns:
        None
    """

    assert isinstance(
        layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    ), f"Can only be applied to linear and conv layers, given {layer.__class__.__name__}"

    initialize_weight(layer.weight, distribution, nonlinearity)
    if layer.bias is not None:
        initialize_bias(
            layer.bias, init_bias, nonlinearity=nonlinearity, weight=layer.weight
        )


def reset_conv2d_parameters(
    model: nn.Module,
    init_distribution: Optional[str],
    init_bias: Optional[float],
    nonlinearity: Optional[str],
) -> None:
    """Reset the parameters of all convolutional layers in the model.

    Args:
        model (nn.Module): The model to reset the convolutional layers of.
        init_distribution (Optional[str]): The distribution to use for weight initialization.
        init_bias (Optional[float]): The value to use for bias initialization.
        nonlinearity (Optional[str]): The type of nonlinearity to use when initializing the layers.

    Returns:
        None
    """

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            initialize_layer(
                module,
                distribution=init_distribution,
                init_bias=init_bias,
                nonlinearity=nonlinearity,
            )
