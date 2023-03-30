import warnings

import torch
from torch import nn
from torch.nn import init


def initialize_weight(weight, distribution, nonlinearity="LeakyReLU"):
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


def initialize_bias(bias, distribution=0, nonlinearity="LeakyReLU", weight=None):
    if distribution is None:
        return
    if isinstance(distribution, (int, float)):
        init.constant_(bias, distribution)
    else:
        raise NotImplementedError(f"Unsupported distribution '{distribution}'")


def initialize_layer(
    layer, distribution="kaiming_normal", init_bias=0, nonlinearity="LeakyReLU"
):

    assert isinstance(
        layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    ), f"Can only be applied to linear and conv layers, given {layer.__class__.__name__}"

    initialize_weight(layer.weight, distribution, nonlinearity)
    if layer.bias is not None:
        initialize_bias(
            layer.bias, init_bias, nonlinearity=nonlinearity, weight=layer.weight
        )


def reset_conv2d_parameters(self, init_distribution, init_bias, nonlinearity):
    for name, module in self.named_modules():
        if isinstance(module, nn.Conv2d):
            initialize_layer(
                module,
                distribution=init_distribution,
                init_bias=init_bias,
                nonlinearity=nonlinearity,
            )
