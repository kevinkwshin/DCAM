import torch
import torch.nn as nn
import torch.nn.functional as F

import monai
from monai.networks.nets import ResNet, DenseNet, SENet
from monai.networks.nets.resnet import ResNetBlock#, ResNetBottleneck
from monai.networks.nets.senet import SEBottleneck, SEResNetBottleneck
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option

from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Type, Union
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, Pool

from .acm import *
from .cbam import *
from .deeprft import *
from .ffc import *
from .nnblock import *
from .scm import *

class SimpleASPP(nn.Module):
    """
    A simplified version of the atrous spatial pyramid pooling (ASPP) module.

    Chen et al., Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.
    https://arxiv.org/abs/1802.02611

    Wang et al., A Noise-robust Framework for Automatic Segmentation of COVID-19 Pneumonia Lesions
    from CT Images. https://ieeexplore.ieee.org/document/9109297
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        conv_out_channels: int,
        kernel_sizes: Sequence[int] = (1, 3, 3, 3),
        dilations: Sequence[int] = (1, 2, 4, 6),
        norm_type: Optional[Union[Tuple, str]] = "INSTANCE",
        # acti_type: Optional[Union[Tuple, str]] = "RELU",
        acti_type: Optional[Union[Tuple, str]] = "GELU",
        bias: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
            in_channels: number of input channels.
            conv_out_channels: number of output channels of each atrous conv.
                The final number of output channels is conv_out_channels * len(kernel_sizes).
            kernel_sizes: a sequence of four convolutional kernel sizes.
                Defaults to (1, 3, 3, 3) for four (dilated) convolutions.
            dilations: a sequence of four convolutional dilation parameters.
                Defaults to (1, 2, 4, 6) for four (dilated) convolutions.
            norm_type: final kernel-size-one convolution normalization type.
                Defaults to batch norm.
            acti_type: final kernel-size-one convolution activation type.
                Defaults to leaky ReLU.
            bias: whether to have a bias term in convolution blocks. Defaults to False.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.

        Raises:
            ValueError: When ``kernel_sizes`` length differs from ``dilations``.

        See also:

            :py:class:`monai.networks.layers.Act`
            :py:class:`monai.networks.layers.Conv`
            :py:class:`monai.networks.layers.Norm`

        """
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError(
                "kernel_sizes and dilations length must match, "
                f"got kernel_sizes={len(kernel_sizes)} dilations={len(dilations)}."
            )
        pads = tuple(same_padding(k, d) for k, d in zip(kernel_sizes, dilations))

        self.convs = nn.ModuleList()
        for k, d, p in zip(kernel_sizes, dilations, pads):
            _conv = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=conv_out_channels, kernel_size=k, dilation=d, padding=p
            )
            # self.convs.append(_conv) # original
            # self.convs.append(nn.Sequential(_conv, DeepRFT(conv_out_channels, conv_out_channels)))
            self.convs.append(nn.Sequential(_conv, ACM(32, conv_out_channels)))
        out_channels = conv_out_channels * len(pads)  # final conv. output channels
        self.conv_k1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            act=acti_type,
            norm=norm_type,
            bias=bias,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape (batch, channel, spatial_1[, spatial_2, ...]).
        """
        # x_out = torch.cat([conv(x) for conv in self.convs], dim=1)
        convs = list()
        for conv in self.convs:
            convs.append(conv(x))
        x_out = torch.cat(convs, dim=1)
        x_out = self.conv_k1(x_out)
        return x_out

class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        aspp = monai.networks.blocks.SimpleASPP(spatial_dims, in_chns, in_chns//4, kernel_sizes= (1, 3, 3, 3), dilations= (1, 2, 4, 6), norm_type=norm, acti_type=act)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        
        self.add_module("max_pooling", max_pooling)
        self.add_module("aspp", aspp)
        self.add_module("convs", convs)
        

class ConvNet(nn.Module):
    def __init__(self, features = [64, 128, 256, 512, 512], spactial_dims = 1, act, norm = 'instance', bias = True, dropout=0.1, module='none'):
        fea = features
        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[0], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_5 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        
    def forward(self,x):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        x5 = self.down_5(x4)
        return x1,x2,x3,x4,x5