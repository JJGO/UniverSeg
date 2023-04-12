from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import einops as E
import torch
from torch import nn

from .nn import CrossConv2d
from .nn import reset_conv2d_parameters
from .nn import Vmap, vmap
from .validation import (Kwargs, as_2tuple, size2t, validate_arguments,
                         validate_arguments_init)


def get_nonlinearity(nonlinearity: Optional[str]) -> nn.Module:
    if nonlinearity is None:
        return nn.Identity()
    if nonlinearity == "Softmax":
        # For Softmax, we need to specify the channel dimension
        return nn.Softmax(dim=1)
    if hasattr(nn, nonlinearity):
        return getattr(nn, nonlinearity)()
    raise ValueError(f"nonlinearity {nonlinearity} not found")


@validate_arguments_init
@dataclass(eq=False, repr=False)
class ConvOp(nn.Sequential):

    in_channels: int
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode="zeros",
            bias=True,
        )

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)

        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )


@validate_arguments_init
@dataclass(eq=False, repr=False)
class CrossOp(nn.Module):

    in_channels: size2t
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()

        self.cross_conv = CrossConv2d(
            in_channels=as_2tuple(self.in_channels),
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)

        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )

    def forward(self, target, support):
        interaction = self.cross_conv(target, support).squeeze(dim=1)

        if self.nonlinearity is not None:
            interaction = vmap(self.nonlin, interaction)

        new_target = interaction.mean(dim=1, keepdims=True)

        return new_target, interaction


@validate_arguments_init
@dataclass(eq=False, repr=False)
class CrossBlock(nn.Module):

    in_channels: size2t
    cross_features: int
    conv_features: Optional[int] = None
    cross_kws: Optional[Dict[str, Any]] = None
    conv_kws: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__init__()

        conv_features = self.conv_features or self.cross_features
        cross_kws = self.cross_kws or {}
        conv_kws = self.conv_kws or {}

        self.cross = CrossOp(self.in_channels, self.cross_features, **cross_kws)
        self.target = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))
        self.support = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))

    def forward(self, target, support):
        target, support = self.cross(target, support)
        target = self.target(target)
        support = self.support(support)
        return target, support


@validate_arguments_init
@dataclass(eq=False, repr=False)
class UniverSeg(nn.Module):

    encoder_blocks: List[size2t]
    decoder_blocks: Optional[List[size2t]] = None

    def __post_init__(self):
        super().__init__()

        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        encoder_blocks = list(map(as_2tuple, self.encoder_blocks))
        decoder_blocks = self.decoder_blocks or encoder_blocks[-2::-1]
        decoder_blocks = list(map(as_2tuple, decoder_blocks))

        block_kws = dict(cross_kws=dict(nonlinearity=None))

        in_ch = (1, 2)
        out_channels = 1
        out_activation = None

        # Encoder
        skip_outputs = []
        for (cross_ch, conv_ch) in encoder_blocks:
            block = CrossBlock(in_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.enc_blocks.append(block)
            skip_outputs.append(in_ch)

        # Decoder
        skip_chs = skip_outputs[-2::-1]
        for (cross_ch, conv_ch), skip_ch in zip(decoder_blocks, skip_chs):
            block = CrossBlock(in_ch + skip_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.dec_blocks.append(block)

        self.out_conv = ConvOp(
            in_ch, out_channels, kernel_size=1, nonlinearity=out_activation,
        )

    def forward(self, target_image, support_images, support_labels):

        target = E.rearrange(target_image, "B 1 H W -> B 1 1 H W")
        support = torch.cat([support_images, support_labels], dim=2)

        pass_through = []

        for i, encoder_block in enumerate(self.enc_blocks):
            target, support = encoder_block(target, support)
            if i == len(self.encoder_blocks) - 1:
                break
            pass_through.append((target, support))
            target = vmap(self.downsample, target)
            support = vmap(self.downsample, support)

        for decoder_block in self.dec_blocks:
            target_skip, support_skip = pass_through.pop()
            target = torch.cat([vmap(self.upsample, target), target_skip], dim=2)
            support = torch.cat([vmap(self.upsample, support), support_skip], dim=2)
            target, support = decoder_block(target, support)

        target = E.rearrange(target, "B 1 C H W -> B C H W")
        target = self.out_conv(target)

        return target


@validate_arguments
def universeg(version: Literal["v1"] = "v1", pretrained: bool = False) -> nn.Module:
    weights = {
        "v1": "https://github.com/JJGO/UniverSeg/releases/download/weights/universeg_v1_nf64_ss64_STA.pt"
    }

    if version == "v1":
        model = UniverSeg(encoder_blocks=[64, 64, 64, 64])

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(weights[version])
        model.load_state_dict(state_dict)

    return model
