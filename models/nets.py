import torch
import torch.nn as nn
import torch.nn.functional as F 

# from .acm import *
# from .cbam import *
# from .deeprft import *
# from .ffc import *
# from .nnblock import *
from .u2net import *
from .unet import *
from .resnet import *

import monai
# from monai.networks.blocks import Convolution, UpSample
# from monai.networks.nets.basic_unet import TwoConv, Down, UpCat, UpSample, Union
# from monai.networks.layers.factories import Act, Conv, Pad, Pool
# from monai.utils import deprecated_arg, ensure_tuple_rep

# from monai.networks.nets import ResNet, DenseNet, SENet
# from monai.networks.nets.resnet import ResNetBlock#, ResNetBottleneck
# from monai.networks.nets.senet import SEBottleneck, SEResNetBottleneck
# from functools import partial
# from typing import Any, Callable, List, Optional, Tuple, Type, Union
# from monai.networks.layers.factories import Act, Conv, Dropout, Norm, Pool

# import math
# import operator
# import re
# from functools import reduce
# from typing import List, NamedTuple, Optional, Tuple, Type, Union

# # from torch.utils import model_zoo
# # import torch.nn.functional as F 

# from monai.networks.layers.utils import get_norm_layer
# from monai.utils.module import look_up_option
# from typing import Optional, Sequence, Union

# from monai.networks.blocks import Convolution, UpSample
# from monai.networks.nets.basic_unet import TwoConv, Down, UpCat, UpSample, Union
# from monai.networks.layers.factories import Conv, Pool
# from monai.utils import deprecated_arg, ensure_tuple_rep

# from monai.networks.nets import ResNet, DenseNet, SENet
# from monai.networks.nets.resnet import ResNetBlock#, ResNetBottleneck
# from monai.networks.nets.senet import SEBottleneck, SEResNetBottleneck

# from functools import partial
# from typing import Any, Callable, List, Optional, Tuple, Type, Union
# from monai.networks.layers.factories import Act, Conv, Dropout, Norm, Pool


# def get_inplanes():
#     return [64, 128, 256, 512]


# def get_avgpool():
#     return [0, 1, (1, 1), (1, 1, 1)]


# class ResNetBlock(nn.Module):
#     expansion = 1

#     def __init__(
#         self,
#         in_planes: int,
#         planes: int,
#         spatial_dims: int = 3,
#         stride: int = 1,
#         downsample: Union[nn.Module, partial, None] = None,
#         module="acm"
#         # module="none"
#     ) -> None:
#         """
#         Args:
#             in_planes: number of input channels.
#             planes: number of output channels.
#             spatial_dims: number of spatial dimensions of the input image.
#             stride: stride to use for first conv layer.
#             downsample: which downsample layer to use.
#         """
#         super().__init__()

#         conv_type: Callable = Conv[Conv.CONV, spatial_dims]
#         # norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
#         norm_type: Callable = Norm[Norm.INSTANCE, spatial_dims]

#         self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
#         self.bn1 = norm_type(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = norm_type(planes)
#         self.downsample = downsample
#         self.stride = stride
        
#         spatial_dims = 1
#         num_acm_groups = 8
#         orthogonal_loss= False
#         # orthogonal_loss= True

#         if module == "none":
#             self.module = None
#         elif module == 'se':
#             self.module = monai.networks.blocks.ResidualSELayer(spatial_dims,plane)
#         elif module =='nlnn':
#             norm = 'instance'
#             self.module = NLBlockND(in_channels=planes, mode='embedded', dimension=spatial_dims, norm_layer=norm)
#         elif module =='deeprft':
#             self.module = FFT_ConvBlock(planes,planes)
#         elif module =='cbam':
#             self.module = CBAM(gate_channels=planes, reduction_ratio=16, pool_types=['avg', 'max'])            
#         elif module == 'acm':
#             self.module = ACM(num_heads=num_acm_groups, num_features=planes, orthogonal_loss=orthogonal_loss)
#             self.module.init_parameters()
#         else:
#             raise ValueError("undefined module")

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         if isinstance(x, tuple):
#             x, prev_dp = x
#         else:
#             prev_dp = None

#         residual = x

#         out: torch.Tensor = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         dp = None
#         if self.module is not None:
#             out = self.module(out)
#             if isinstance(out, tuple):
#                 out, dp = out
#                 if prev_dp is not None:
#                     dp = prev_dp + dp

#         out += residual
#         out = self.relu(out)

#         return out


# class ResNetBottleneck(nn.Module):
#     expansion = 4

#     def __init__(
#         self,
#         in_planes: int,
#         planes: int,
#         spatial_dims: int = 3,
#         stride: int = 1,
#         downsample: Union[nn.Module, partial, None] = None,
#         module="acm"
#         # module="none"
#     ) -> None:
#         """
#         Args:
#             in_planes: number of input channels.
#             planes: number of output channels (taking expansion into account).
#             spatial_dims: number of spatial dimensions of the input image.
#             stride: stride to use for second conv layer.
#             downsample: which downsample layer to use.
#         """

#         super().__init__()

#         conv_type: Callable = Conv[Conv.CONV, spatial_dims]
#         # norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
#         norm_type: Callable = Norm[Norm.INSTANCE, spatial_dims]

#         self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = norm_type(planes)
#         self.conv2 = conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = norm_type(planes)
#         self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = norm_type(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#         num_acm_groups = 8
#         orthogonal_loss= False
#         orthogonal_loss= True

#         if module == "none":
#             self.module = None
#         elif module == 'acm':
#             self.module = ACM(num_heads=num_acm_groups, num_features=planes * 4, orthogonal_loss=orthogonal_loss)
#             self.module.init_parameters()
#         else:
#             raise ValueError("undefined module")


#     def forward(self, x: torch.Tensor) -> torch.Tensor:
        
#         if isinstance(x, tuple):
#             x, prev_dp = x
#         else:
#             prev_dp = None

#         residual = x

#         out: torch.Tensor = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)
            
#         dp = None
#         if self.module is not None:
#             out = self.module(out)
#             if isinstance(out, tuple):
#                 out, dp = out
#                 if prev_dp is not None:
#                     dp = prev_dp + dp

#         out += residual
#         out = self.relu(out)

#         if dp is None:
#             return out
#         else:
#             # diff loss
#             return out, dp

# # ResNet = monai.networks.nets.ResNet
# class ResNetFeature(ResNet):
#     def __init__(
#             self,
#             block: Union[Type[Union[ResNetBlock, ResNetBottleneck]], str],
#             layers: List[int],
#             block_inplanes: List[int],
#             spatial_dims: int = 3,
#             n_input_channels: int = 3,
#             conv1_t_size: Union[Tuple[int], int] = 7,
#             conv1_t_stride: Union[Tuple[int], int] = 1,
#             no_max_pool: bool = False,
#             shortcut_type: str = "B",
#             widen_factor: float = 1.0,
#             num_classes: int = 400,
#             feed_forward: bool = True,
#             n_classes: Optional[int] = None,
#         ) -> None:
#         super().__init__(block,layers,block_inplanes,spatial_dims,n_input_channels,conv1_t_size,conv1_t_stride,no_max_pool)
#         self.spatial_dims= spatial_dims
    
#         pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
#         self.mpool1 = pool_type(kernel_size=3, stride=2, ceil_mode=True)
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x1 = self.conv1(x)
#         x1 = self.bn1(x1)
#         x1 = self.relu(x1)
#         if not self.no_max_pool:
#             x1 = self.maxpool(x1)
#         x2 = self.layer1(x1)        
#         x2 = self.mpool1(x2)
#         x3 = self.layer2(x2)
#         x4 = self.layer3(x3)
#         x5 = self.layer4(x4)
#         return x1, x2, x3, x4, x5

# def _resnet(
#     arch: str,
#     block: Type[Union[ResNetBlock, ResNetBottleneck]],
#     layers: List[int],
#     block_inplanes: List[int],
#     pretrained: bool,
#     progress: bool,
#     **kwargs: Any,
# ) -> ResNet:
#     model: ResNetFeature = ResNetFeature(block, layers, block_inplanes, **kwargs)
#     if pretrained:
#         # Author of paper zipped the state_dict on googledrive,
#         # so would need to download, unzip and read (2.8gb file for a ~150mb state dict).
#         # Would like to load dict from url but need somewhere to save the state dicts.
#         raise NotImplementedError(
#             "Currently not implemented. You need to manually download weights provided by the paper's author"
#             " and load then to the model with `state_dict`. See https://github.com/Tencent/MedicalNet"
#         )
#     return model


# def resnet10(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-10 with optional pretrained support when `spatial_dims` is 3.

#     Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet("resnet10", ResNetBlock, [1, 1, 1, 1], get_inplanes(), pretrained, progress, **kwargs)


# def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-18 with optional pretrained support when `spatial_dims` is 3.

#     Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet("resnet18", ResNetBlock, [2, 2, 2, 2], get_inplanes(), pretrained, progress, **kwargs)


# def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-34 with optional pretrained support when `spatial_dims` is 3.

#     Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet("resnet34", ResNetBlock, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)


# def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-50 with optional pretrained support when `spatial_dims` is 3.

#     Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet("resnet50", ResNetBottleneck, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)

def bn2instance(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.InstanceNorm1d(module.num_features,
                                                module.eps, module.momentum,
                                                module.affine,
                                                module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, bn2instance(child))

    del module
    return module_output

def _upsample_like(src,tar):
    # src = F.upsample(src,size=tar.shape[2:], mode='linear')
    src = F.upsample(src,size=tar.shape[2:], mode='nearest')
    return src

# import math
# import operator
# import re
# from functools import reduce
# from typing import List, NamedTuple, Optional, Tuple, Type, Union

# import torch
# from torch import nn
# from torch.utils import model_zoo

# from monai.networks.blocks import Convolution
# from monai.networks.layers.factories import Act, Conv, Pad, Pool
# from monai.networks.layers.utils import get_norm_layer
# from monai.utils.module import look_up_option

# __all__ = [
#     "EfficientNet",
#     "EfficientNetBN",
#     "get_efficientnet_image_size",
#     "drop_connect",
#     "EfficientNetBNFeatures",
#     "BlockArgs",
# ]

# efficientnet_params = {
#     # model_name: (width_mult, depth_mult, image_size, dropout_rate, dropconnect_rate)
#     "efficientnet-b0": (1.0, 1.0, 224, 0.2, 0.2),
#     "efficientnet-b1": (1.0, 1.1, 240, 0.2, 0.2),
#     "efficientnet-b2": (1.1, 1.2, 260, 0.3, 0.2),
#     "efficientnet-b3": (1.2, 1.4, 300, 0.3, 0.2),
#     "efficientnet-b4": (1.4, 1.8, 380, 0.4, 0.2),
#     "efficientnet-b5": (1.6, 2.2, 456, 0.4, 0.2),
#     "efficientnet-b6": (1.8, 2.6, 528, 0.5, 0.2),
#     "efficientnet-b7": (2.0, 3.1, 600, 0.5, 0.2),
#     "efficientnet-b8": (2.2, 3.6, 672, 0.5, 0.2),
#     "efficientnet-l2": (4.3, 5.3, 800, 0.5, 0.2),
# }

# url_map = {
#     "efficientnet-b0": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
#     "efficientnet-b1": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
#     "efficientnet-b2": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
#     "efficientnet-b3": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
#     "efficientnet-b4": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
#     "efficientnet-b5": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
#     "efficientnet-b6": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
#     "efficientnet-b7": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth",
#     # trained with adversarial examples, simplify the name to decrease string length
#     "b0-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth",
#     "b1-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth",
#     "b2-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth",
#     "b3-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth",
#     "b4-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth",
#     "b5-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth",
#     "b6-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth",
#     "b7-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth",
#     "b8-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth",
# }

# class MBConvBlock(nn.Module):
#     def __init__(
#         self,
#         spatial_dims: int,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int,
#         stride: int,
#         image_size: List[int],
#         expand_ratio: int,
#         se_ratio: Optional[float],
#         id_skip: Optional[bool] = True,
#         norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
#         drop_connect_rate: Optional[float] = 0.2,
#         se_module='se',
#     ) -> None:
#         """
#         Mobile Inverted Residual Bottleneck Block.

#         Args:
#             spatial_dims: number of spatial dimensions.
#             in_channels: number of input channels.
#             out_channels: number of output channels.
#             kernel_size: size of the kernel for conv ops.
#             stride: stride to use for conv ops.
#             image_size: input image resolution.
#             expand_ratio: expansion ratio for inverted bottleneck.
#             se_ratio: squeeze-excitation ratio for se layers.
#             id_skip: whether to use skip connection.
#             norm: feature normalization type and arguments. Defaults to batch norm.
#             drop_connect_rate: dropconnect rate for drop connection (individual weights) layers.

#         References:
#             [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
#             [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
#             [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
#         """
#         super().__init__()

#         # select the type of N-Dimensional layers to use
#         # these are based on spatial dims and selected from MONAI factories
#         conv_type = Conv["conv", spatial_dims]
#         adaptivepool_type = Pool["adaptiveavg", spatial_dims]

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.id_skip = id_skip
#         self.stride = stride
#         self.expand_ratio = expand_ratio
#         self.drop_connect_rate = drop_connect_rate
#         self.se_module = se_module
        
#         if (se_ratio is not None) and (0.0 < se_ratio <= 1.0):
#             self.has_se = True
#             self.se_ratio = se_ratio
#         else:
#             self.has_se = False

#         # Expansion phase (Inverted Bottleneck)
#         inp = in_channels  # number of input channels
#         oup = in_channels * expand_ratio  # number of output channels
#         if self.expand_ratio != 1:
#             self._expand_conv = conv_type(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
#             self._expand_conv_padding = _make_same_padder(self._expand_conv, image_size)

#             self._bn0 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=oup)
#         else:
#             # need to have the following to fix JIT error:
#             #   "Module 'MBConvBlock' has no attribute '_expand_conv'"

#             # FIXME: find a better way to bypass JIT error
#             self._expand_conv = nn.Identity()
#             self._expand_conv_padding = nn.Identity()
#             self._bn0 = nn.Identity()

#         # Depthwise convolution phase
#         self._depthwise_conv = conv_type(
#             in_channels=oup,
#             out_channels=oup,
#             groups=oup,  # groups makes it depthwise
#             kernel_size=kernel_size,
#             stride=self.stride,
#             bias=False,
#         )
#         self._depthwise_conv_padding = _make_same_padder(self._depthwise_conv, image_size)
#         self._bn1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=oup)
#         image_size = _calculate_output_image_size(image_size, self.stride)

#         # Squeeze and Excitation layer, if desired
#         if self.has_se:
#             if se_module == 'se':
#                 self._se_adaptpool = adaptivepool_type(1)
#                 num_squeezed_channels = max(1, int(in_channels * self.se_ratio))
#                 self._se_reduce = conv_type(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
#                 self._se_reduce_padding = _make_same_padder(self._se_reduce, [1, 1])
#                 self._se_expand = conv_type(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
#                 self._se_expand_padding = _make_same_padder(self._se_expand, [1, 1])
#             elif se_module =='acm':
#                 num_acm_groups = 4
#                 orthogonal_loss= False
#                 self.se = ACM(num_heads=num_acm_groups, num_features=oup, orthogonal_loss=orthogonal_loss)
#             elif se_module =='nlnn':
#                 self.se = NLBlockND(in_channels=oup, mode='embedded', dimension=spatial_dims, norm_layer=norm)
#             elif se_module =='deeprft':
#                 self.se = FFT_ConvBlock(oup,oup)
#             elif se_module =='cbam':
#                 self.se = CBAM(gate_channels=oup, reduction_ratio=16, pool_types=['avg', 'max'])
                
#         # Pointwise convolution phase
#         final_oup = out_channels
#         self._project_conv = conv_type(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
#         self._project_conv_padding = _make_same_padder(self._project_conv, image_size)
#         self._bn2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=final_oup)

#         # swish activation to use - using memory efficient swish by default
#         # can be switched to normal swish using self.set_swish() function call
#         self._swish = Act["memswish"](inplace=True)

#     def forward(self, inputs: torch.Tensor):
#         """MBConvBlock"s forward function.

#         Args:
#             inputs: Input tensor.

#         Returns:
#             Output of this block after processing.
#         """
#         # Expansion and Depthwise Convolution
#         x = inputs
#         if self.expand_ratio != 1:
#             x = self._expand_conv(self._expand_conv_padding(x))
#             x = self._bn0(x)
#             x = self._swish(x)

#         x = self._depthwise_conv(self._depthwise_conv_padding(x))
#         x = self._bn1(x)
#         x = self._swish(x)

#         # Squeeze and Excitation
#         if self.has_se:
#             if self.se_module =='se':
#                 x_squeezed = self._se_adaptpool(x)
#                 x_squeezed = self._se_reduce(self._se_reduce_padding(x_squeezed))
#                 x_squeezed = self._swish(x_squeezed)
#                 x_squeezed = self._se_expand(self._se_expand_padding(x_squeezed))
#                 x = torch.sigmoid(x_squeezed) * x
#             else:
#                 x = self.se(x)

#         # Pointwise Convolution
#         x = self._project_conv(self._project_conv_padding(x))
#         x = self._bn2(x)

#         # Skip connection and drop connect
#         if self.id_skip and self.stride == 1 and self.in_channels == self.out_channels:
#             # the combination of skip connection and drop connect brings about stochastic depth.
#             if self.drop_connect_rate:
#                 x = drop_connect(x, p=self.drop_connect_rate, training=self.training)
#             x = x + inputs  # skip connection
#         return x

#     def set_swish(self, memory_efficient: bool = True) -> None:
#         """Sets swish function as memory efficient (for training) or standard (for export).

#         Args:
#             memory_efficient (bool): Whether to use memory-efficient version of swish.
#         """
#         self._swish = Act["memswish"](inplace=True) if memory_efficient else Act["swish"](alpha=1.0)


# class EfficientNet(nn.Module):
#     def __init__(
#         self,
#         blocks_args_str: List[str],
#         spatial_dims: int = 2,
#         in_channels: int = 3,
#         num_classes: int = 1000,
#         width_coefficient: float = 1.0,
#         depth_coefficient: float = 1.0,
#         dropout_rate: float = 0.2,
#         image_size: int = 224,
#         norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
#         drop_connect_rate: float = 0.2,
#         depth_divisor: int = 8,
#         se_module='se'
#     ) -> None:
#         """
#         EfficientNet based on `Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/pdf/1905.11946.pdf>`_.
#         Adapted from `EfficientNet-PyTorch <https://github.com/lukemelas/EfficientNet-PyTorch>`_.

#         Args:
#             blocks_args_str: block definitions.
#             spatial_dims: number of spatial dimensions.
#             in_channels: number of input channels.
#             num_classes: number of output classes.
#             width_coefficient: width multiplier coefficient (w in paper).
#             depth_coefficient: depth multiplier coefficient (d in paper).
#             dropout_rate: dropout rate for dropout layers.
#             image_size: input image resolution.
#             norm: feature normalization type and arguments.
#             drop_connect_rate: dropconnect rate for drop connection (individual weights) layers.
#             depth_divisor: depth divisor for channel rounding.

#         """
#         super().__init__()

#         if spatial_dims not in (1, 2, 3):
#             raise ValueError("spatial_dims can only be 1, 2 or 3.")

#         # select the type of N-Dimensional layers to use
#         # these are based on spatial dims and selected from MONAI factories
#         conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv["conv", spatial_dims]
#         adaptivepool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
#             "adaptiveavg", spatial_dims
#         ]

#         # decode blocks args into arguments for MBConvBlock
#         blocks_args = [BlockArgs.from_string(s) for s in blocks_args_str]

#         # checks for successful decoding of blocks_args_str
#         if not isinstance(blocks_args, list):
#             raise ValueError("blocks_args must be a list")

#         if blocks_args == []:
#             raise ValueError("block_args must be non-empty")

#         self._blocks_args = blocks_args
#         self.num_classes = num_classes
#         self.in_channels = in_channels
#         self.drop_connect_rate = drop_connect_rate

#         # expand input image dimensions to list
#         current_image_size = [image_size] * spatial_dims

#         # Stem
#         stride = 2
#         out_channels = _round_filters(32, width_coefficient, depth_divisor)  # number of output channels
#         self._conv_stem = conv_type(self.in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
#         self._conv_stem_padding = _make_same_padder(self._conv_stem, current_image_size)
#         self._bn0 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels)
#         current_image_size = _calculate_output_image_size(current_image_size, stride)

#         # build MBConv blocks
#         num_blocks = 0
#         self._blocks = nn.Sequential()

#         self.extract_stacks = []

#         # update baseline blocks to input/output filters and number of repeats based on width and depth multipliers.
#         for idx, block_args in enumerate(self._blocks_args):
#             block_args = block_args._replace(
#                 input_filters=_round_filters(block_args.input_filters, width_coefficient, depth_divisor),
#                 output_filters=_round_filters(block_args.output_filters, width_coefficient, depth_divisor),
#                 num_repeat=_round_repeats(block_args.num_repeat, depth_coefficient),
#             )
#             self._blocks_args[idx] = block_args

#             # calculate the total number of blocks - needed for drop_connect estimation
#             num_blocks += block_args.num_repeat

#             if block_args.stride > 1:
#                 self.extract_stacks.append(idx)

#         self.extract_stacks.append(len(self._blocks_args))

#         # create and add MBConvBlocks to self._blocks
#         idx = 0  # block index counter
#         for stack_idx, block_args in enumerate(self._blocks_args):
#             blk_drop_connect_rate = self.drop_connect_rate

#             # scale drop connect_rate
#             if blk_drop_connect_rate:
#                 blk_drop_connect_rate *= float(idx) / num_blocks

#             sub_stack = nn.Sequential()
#             # the first block needs to take care of stride and filter size increase.
#             sub_stack.add_module(
#                 str(idx),
#                 MBConvBlock(
#                     spatial_dims=spatial_dims,
#                     in_channels=block_args.input_filters,
#                     out_channels=block_args.output_filters,
#                     kernel_size=block_args.kernel_size,
#                     stride=block_args.stride,
#                     image_size=current_image_size,
#                     expand_ratio=block_args.expand_ratio,
#                     se_ratio=block_args.se_ratio,
#                     id_skip=block_args.id_skip,
#                     norm=norm,
#                     drop_connect_rate=blk_drop_connect_rate,
#                     se_module=se_module
#                 ),
#             )
#             idx += 1  # increment blocks index counter

#             current_image_size = _calculate_output_image_size(current_image_size, block_args.stride)
#             if block_args.num_repeat > 1:  # modify block_args to keep same output size
#                 block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)

#             # add remaining block repeated num_repeat times
#             for _ in range(block_args.num_repeat - 1):
#                 blk_drop_connect_rate = self.drop_connect_rate

#                 # scale drop connect_rate
#                 if blk_drop_connect_rate:
#                     blk_drop_connect_rate *= float(idx) / num_blocks

#                 # add blocks
#                 sub_stack.add_module(
#                     str(idx),
#                     MBConvBlock(
#                         spatial_dims=spatial_dims,
#                         in_channels=block_args.input_filters,
#                         out_channels=block_args.output_filters,
#                         kernel_size=block_args.kernel_size,
#                         stride=block_args.stride,
#                         image_size=current_image_size,
#                         expand_ratio=block_args.expand_ratio,
#                         se_ratio=block_args.se_ratio,
#                         id_skip=block_args.id_skip,
#                         norm=norm,
#                         drop_connect_rate=blk_drop_connect_rate,
#                         se_module=se_module
#                     ),
#                 )
#                 idx += 1  # increment blocks index counter

#             self._blocks.add_module(str(stack_idx), sub_stack)

#         # sanity check to see if len(self._blocks) equal expected num_blocks
#         if idx != num_blocks:
#             raise ValueError("total number of blocks created != num_blocks")

#         # Head
#         head_in_channels = block_args.output_filters
#         out_channels = _round_filters(1280, width_coefficient, depth_divisor)
#         self._conv_head = conv_type(head_in_channels, out_channels, kernel_size=1, bias=False)
#         self._conv_head_padding = _make_same_padder(self._conv_head, current_image_size)
#         self._bn1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels)

#         # final linear layer
#         self._avg_pooling = adaptivepool_type(1)
#         self._dropout = nn.Dropout(dropout_rate)
#         self._fc = nn.Linear(out_channels, self.num_classes)

#         # swish activation to use - using memory efficient swish by default
#         # can be switched to normal swish using self.set_swish() function call
#         self._swish = Act["memswish"]()

#         # initialize weights using Tensorflow's init method from official impl.
#         self._initialize_weights()


#     def set_swish(self, memory_efficient: bool = True) -> None:
#         """
#         Sets swish function as memory efficient (for training) or standard (for JIT export).

#         Args:
#             memory_efficient: whether to use memory-efficient version of swish.

#         """
#         self._swish = Act["memswish"]() if memory_efficient else Act["swish"](alpha=1.0)
#         for sub_stack in self._blocks:
#             for block in sub_stack:
#                 block.set_swish(memory_efficient)


#     def forward(self, inputs: torch.Tensor):
#         """
#         Args:
#             inputs: input should have spatially N dimensions
#             ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.

#         Returns:
#             a torch Tensor of classification prediction in shape ``(Batch, num_classes)``.
#         """
#         # Stem
#         x = self._conv_stem(self._conv_stem_padding(inputs))
#         x = self._swish(self._bn0(x))
#         # Blocks
#         x = self._blocks(x)
#         # Head
#         x = self._conv_head(self._conv_head_padding(x))
#         x = self._swish(self._bn1(x))

#         # Pooling and final linear layer
#         x = self._avg_pooling(x)

#         x = x.flatten(start_dim=1)
#         x = self._dropout(x)
#         x = self._fc(x)
#         return x


#     def _initialize_weights(self) -> None:
#         """
#         Args:
#             None, initializes weights for conv/linear/batchnorm layers
#             following weight init methods from
#             `official Tensorflow EfficientNet implementation
#             <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py#L61>`_.
#             Adapted from `EfficientNet-PyTorch's init method
#             <https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/efficientnet_builder.py>`_.
#         """
#         for _, m in self.named_modules():
#             if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
#                 fan_out = reduce(operator.mul, m.kernel_size, 1) * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
#                 m.weight.data.fill_(1.0)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 fan_out = m.weight.size(0)
#                 fan_in = 0
#                 init_range = 1.0 / math.sqrt(fan_in + fan_out)
#                 m.weight.data.uniform_(-init_range, init_range)
#                 m.bias.data.zero_()



# class EfficientNetBN(EfficientNet):
#     def __init__(
#         self,
#         model_name: str,
#         pretrained: bool = True,
#         progress: bool = True,
#         spatial_dims: int = 2,
#         in_channels: int = 3,
#         num_classes: int = 1000,
#         norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
#         adv_prop: bool = False,
#     ) -> None:
#         """
#         Generic wrapper around EfficientNet, used to initialize EfficientNet-B0 to EfficientNet-B7 models
#         model_name is mandatory argument as there is no EfficientNetBN itself,
#         it needs the N in [0, 1, 2, 3, 4, 5, 6, 7, 8] to be a model

#         Args:
#             model_name: name of model to initialize, can be from [efficientnet-b0, ..., efficientnet-b8, efficientnet-l2].
#             pretrained: whether to initialize pretrained ImageNet weights, only available for spatial_dims=2 and batch
#                 norm is used.
#             progress: whether to show download progress for pretrained weights download.
#             spatial_dims: number of spatial dimensions.
#             in_channels: number of input channels.
#             num_classes: number of output classes.
#             norm: feature normalization type and arguments.
#             adv_prop: whether to use weights trained with adversarial examples.
#                 This argument only works when `pretrained` is `True`.

#         Examples::

#             # for pretrained spatial 2D ImageNet
#             >>> image_size = get_efficientnet_image_size("efficientnet-b0")
#             >>> inputs = torch.rand(1, 3, image_size, image_size)
#             >>> model = EfficientNetBN("efficientnet-b0", pretrained=True)
#             >>> model.eval()
#             >>> outputs = model(inputs)

#             # create spatial 2D
#             >>> model = EfficientNetBN("efficientnet-b0", spatial_dims=2)

#             # create spatial 3D
#             >>> model = EfficientNetBN("efficientnet-b0", spatial_dims=3)

#             # create EfficientNetB7 for spatial 2D
#             >>> model = EfficientNetBN("efficientnet-b7", spatial_dims=2)

#         """
#         # block args
#         blocks_args_str = [
#             "r1_k3_s11_e1_i32_o16_se0.25",
#             "r2_k3_s22_e6_i16_o24_se0.25",
#             "r2_k5_s22_e6_i24_o40_se0.25",
#             "r3_k3_s22_e6_i40_o80_se0.25",
#             "r3_k5_s11_e6_i80_o112_se0.25",
#             "r4_k5_s22_e6_i112_o192_se0.25",
#             "r1_k3_s11_e6_i192_o320_se0.25",
#         ]

#         # check if model_name is valid model
#         if model_name not in efficientnet_params.keys():
#             raise ValueError(
#                 "invalid model_name {} found, must be one of {} ".format(
#                     model_name, ", ".join(efficientnet_params.keys())
#                 )
#             )

#         # get network parameters
#         weight_coeff, depth_coeff, image_size, dropout_rate, dropconnect_rate = efficientnet_params[model_name]

#         # create model and initialize random weights
#         super().__init__(
#             blocks_args_str=blocks_args_str,
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             num_classes=num_classes,
#             width_coefficient=weight_coeff,
#             depth_coefficient=depth_coeff,
#             dropout_rate=dropout_rate,
#             image_size=image_size,
#             drop_connect_rate=dropconnect_rate,
#             norm=norm,
#         )

#         # only pretrained for when `spatial_dims` is 2
#         if pretrained and (spatial_dims == 2):
#             _load_state_dict(self, model_name, progress, adv_prop)


# # EfficientNet = monai.networks.nets.EfficientNet
# class EfficientNetBNFeatures(EfficientNet):
#     def __init__(
#         self,
#         model_name: str,
#         pretrained: bool = True,
#         progress: bool = True,
#         spatial_dims: int = 2,
#         in_channels: int = 3,
#         num_classes: int = 1000,
#         norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
#         adv_prop: bool = False,
#         se_module='se'
#     ) -> None:
#         """
#         Initialize EfficientNet-B0 to EfficientNet-B7 models as a backbone, the backbone can
#         be used as an encoder for segmentation and objection models.
#         Compared with the class `EfficientNetBN`, the only different place is the forward function.

#         This class refers to `PyTorch image models <https://github.com/rwightman/pytorch-image-models>`_.

#         """
#         blocks_args_str = [
#             "r1_k3_s11_e1_i32_o16_se0.25",
#             "r2_k3_s22_e6_i16_o24_se0.25",
#             "r2_k5_s22_e6_i24_o40_se0.25",
#             "r3_k3_s22_e6_i40_o80_se0.25",
#             "r3_k5_s11_e6_i80_o112_se0.25",
#             "r4_k5_s22_e6_i112_o192_se0.25",
#             "r1_k3_s11_e6_i192_o320_se0.25",
#         ]

#         # check if model_name is valid model
#         if model_name not in efficientnet_params.keys():
#             raise ValueError(
#                 "invalid model_name {} found, must be one of {} ".format(
#                     model_name, ", ".join(efficientnet_params.keys())
#                 )
#             )

#         # get network parameters
#         weight_coeff, depth_coeff, image_size, dropout_rate, dropconnect_rate = efficientnet_params[model_name]

#         # create model and initialize random weights
#         super().__init__(
#             blocks_args_str=blocks_args_str,
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             num_classes=num_classes,
#             width_coefficient=weight_coeff,
#             depth_coefficient=depth_coeff,
#             dropout_rate=dropout_rate,
#             image_size=image_size,
#             drop_connect_rate=dropconnect_rate,
#             norm=norm,
#             se_module=se_module
#         )

#         # only pretrained for when `spatial_dims` is 2
#         if pretrained and (spatial_dims == 2):
#             _load_state_dict(self, model_name, progress, adv_prop)


#     def forward(self, inputs: torch.Tensor):
#         """
#         Args:
#             inputs: input should have spatially N dimensions
#             ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.

#         Returns:
#             a list of torch Tensors.
#         """
#         # Stem
#         x = self._conv_stem(self._conv_stem_padding(inputs))
#         x = self._swish(self._bn0(x))

#         features = []
#         if 0 in self.extract_stacks:
#             features.append(x)
#         for i, block in enumerate(self._blocks):
#             x = block(x)
#             if i + 1 in self.extract_stacks:
#                 features.append(x)
#         return features



# def get_efficientnet_image_size(model_name: str) -> int:
#     """
#     Get the input image size for a given efficientnet model.

#     Args:
#         model_name: name of model to initialize, can be from [efficientnet-b0, ..., efficientnet-b7].

#     Returns:
#         Image size for single spatial dimension as integer.

#     """
#     # check if model_name is valid model
#     if model_name not in efficientnet_params.keys():
#         raise ValueError(
#             "invalid model_name {} found, must be one of {} ".format(model_name, ", ".join(efficientnet_params.keys()))
#         )

#     # return input image size (all dims equal so only need to return for one dim)
#     _, _, res, _, _ = efficientnet_params[model_name]
#     return res


# def drop_connect(inputs: torch.Tensor, p: float, training: bool) -> torch.Tensor:
#     """
#     Drop connect layer that drops individual connections.
#     Differs from dropout as dropconnect drops connections instead of whole neurons as in dropout.

#     Based on `Deep Networks with Stochastic Depth <https://arxiv.org/pdf/1603.09382.pdf>`_.
#     Adapted from `Official Tensorflow EfficientNet utils
#     <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/utils.py>`_.

#     This function is generalized for MONAI's N-Dimensional spatial activations
#     e.g. 1D activations [B, C, H], 2D activations [B, C, H, W] and 3D activations [B, C, H, W, D]

#     Args:
#         inputs: input tensor with [B, C, dim_1, dim_2, ..., dim_N] where N=spatial_dims.
#         p: probability to use for dropping connections.
#         training: whether in training or evaluation mode.

#     Returns:
#         output: output tensor after applying drop connection.
#     """
#     if p < 0.0 or p > 1.0:
#         raise ValueError(f"p must be in range of [0, 1], found {p}")

#     # eval mode: drop_connect is switched off - so return input without modifying
#     if not training:
#         return inputs

#     # train mode: calculate and apply drop_connect
#     batch_size: int = inputs.shape[0]
#     keep_prob: float = 1 - p
#     num_dims: int = len(inputs.shape) - 2

#     # build dimensions for random tensor, use num_dims to populate appropriate spatial dims
#     random_tensor_shape: List[int] = [batch_size, 1] + [1] * num_dims

#     # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
#     random_tensor: torch.Tensor = torch.rand(random_tensor_shape, dtype=inputs.dtype, device=inputs.device)
#     random_tensor += keep_prob

#     # round to form binary tensor
#     binary_tensor: torch.Tensor = torch.floor(random_tensor)

#     # drop connect using binary tensor
#     output: torch.Tensor = inputs / keep_prob * binary_tensor
#     return output


# def _load_state_dict(model: nn.Module, arch: str, progress: bool, adv_prop: bool) -> None:
#     if adv_prop:
#         arch = arch.split("efficientnet-")[-1] + "-ap"
#     model_url = look_up_option(arch, url_map, None)
#     if model_url is None:
#         print(f"pretrained weights of {arch} is not provided")
#     else:
#         # load state dict from url
#         model_url = url_map[arch]
#         pretrain_state_dict = model_zoo.load_url(model_url, progress=progress)
#         model_state_dict = model.state_dict()

#         pattern = re.compile(r"(.+)\.\d+(\.\d+\..+)")
#         for key, value in model_state_dict.items():
#             pretrain_key = re.sub(pattern, r"\1\2", key)
#             if pretrain_key in pretrain_state_dict and value.shape == pretrain_state_dict[pretrain_key].shape:
#                 model_state_dict[key] = pretrain_state_dict[pretrain_key]

#         model.load_state_dict(model_state_dict)


# def _get_same_padding_conv_nd(
#     image_size: List[int], kernel_size: Tuple[int, ...], dilation: Tuple[int, ...], stride: Tuple[int, ...]
# ) -> List[int]:
#     """
#     Helper for getting padding (nn.ConstantPadNd) to be used to get SAME padding
#     conv operations similar to Tensorflow's SAME padding.

#     This function is generalized for MONAI's N-Dimensional spatial operations (e.g. Conv1D, Conv2D, Conv3D)

#     Args:
#         image_size: input image/feature spatial size.
#         kernel_size: conv kernel's spatial size.
#         dilation: conv dilation rate for Atrous conv.
#         stride: stride for conv operation.

#     Returns:
#         paddings for ConstantPadNd padder to be used on input tensor to conv op.
#     """
#     # get number of spatial dimensions, corresponds to kernel size length
#     num_dims = len(kernel_size)

#     # additional checks to populate dilation and stride (in case they are single entry tuples)
#     if len(dilation) == 1:
#         dilation = dilation * num_dims

#     if len(stride) == 1:
#         stride = stride * num_dims

#     # equation to calculate (pad^+ + pad^-) size
#     _pad_size: List[int] = [
#         max((math.ceil(_i_s / _s) - 1) * _s + (_k_s - 1) * _d + 1 - _i_s, 0)
#         for _i_s, _k_s, _d, _s in zip(image_size, kernel_size, dilation, stride)
#     ]
#     # distribute paddings into pad^+ and pad^- following Tensorflow's same padding strategy
#     _paddings: List[Tuple[int, int]] = [(_p // 2, _p - _p // 2) for _p in _pad_size]

#     # unroll list of tuples to tuples, and then to list
#     # reversed as nn.ConstantPadNd expects paddings starting with last dimension
#     _paddings_ret: List[int] = [outer for inner in reversed(_paddings) for outer in inner]
#     return _paddings_ret


# def _make_same_padder(conv_op: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], image_size: List[int]):
#     """
#     Helper for initializing ConstantPadNd with SAME padding similar to Tensorflow.
#     Uses output of _get_same_padding_conv_nd() to get the padding size.

#     This function is generalized for MONAI's N-Dimensional spatial operations (e.g. Conv1D, Conv2D, Conv3D)

#     Args:
#         conv_op: nn.ConvNd operation to extract parameters for op from
#         image_size: input image/feature spatial size

#     Returns:
#         If padding required then nn.ConstandNd() padder initialized to paddings otherwise nn.Identity()
#     """
#     # calculate padding required
#     padding: List[int] = _get_same_padding_conv_nd(image_size, conv_op.kernel_size, conv_op.dilation, conv_op.stride)

#     # initialize and return padder
#     padder = Pad["constantpad", len(padding) // 2]
#     if sum(padding) > 0:
#         return padder(padding=padding, value=0.0)
#     return nn.Identity()


# def _round_filters(filters: int, width_coefficient: Optional[float], depth_divisor: float) -> int:
#     """
#     Calculate and round number of filters based on width coefficient multiplier and depth divisor.

#     Args:
#         filters: number of input filters.
#         width_coefficient: width coefficient for model.
#         depth_divisor: depth divisor to use.

#     Returns:
#         new_filters: new number of filters after calculation.
#     """

#     if not width_coefficient:
#         return filters

#     multiplier: float = width_coefficient
#     divisor: float = depth_divisor
#     filters_float: float = filters * multiplier

#     # follow the formula transferred from official TensorFlow implementation
#     new_filters: float = max(divisor, int(filters_float + divisor / 2) // divisor * divisor)
#     if new_filters < 0.9 * filters_float:  # prevent rounding by more than 10%
#         new_filters += divisor
#     return int(new_filters)


# def _round_repeats(repeats: int, depth_coefficient: Optional[float]) -> int:
#     """
#     Re-calculate module's repeat number of a block based on depth coefficient multiplier.

#     Args:
#         repeats: number of original repeats.
#         depth_coefficient: depth coefficient for model.

#     Returns:
#         new repeat: new number of repeat after calculating.
#     """
#     if not depth_coefficient:
#         return repeats

#     # follow the formula transferred from official TensorFlow impl.
#     return int(math.ceil(depth_coefficient * repeats))


# def _calculate_output_image_size(input_image_size: List[int], stride: Union[int, Tuple[int]]):
#     """
#     Calculates the output image size when using _make_same_padder with a stride.
#     Required for static padding.

#     Args:
#         input_image_size: input image/feature spatial size.
#         stride: Conv2d operation"s stride.

#     Returns:
#         output_image_size: output image/feature spatial size.
#     """

#     # checks to extract integer stride in case tuple was received
#     if isinstance(stride, tuple):
#         all_strides_equal = all(stride[0] == s for s in stride)
#         if not all_strides_equal:
#             raise ValueError(f"unequal strides are not possible, got {stride}")

#         stride = stride[0]

#     # return output image size
#     return [int(math.ceil(im_sz / stride)) for im_sz in input_image_size]


# class BlockArgs(NamedTuple):
#     """
#     BlockArgs object to assist in decoding string notation
#         of arguments for MBConvBlock definition.
#     """

#     num_repeat: int
#     kernel_size: int
#     stride: int
#     expand_ratio: int
#     input_filters: int
#     output_filters: int
#     id_skip: bool
#     se_ratio: Optional[float] = None

#     @staticmethod
#     def from_string(block_string: str):
#         """
#         Get a BlockArgs object from a string notation of arguments.

#         Args:
#             block_string (str): A string notation of arguments.
#                                 Examples: "r1_k3_s11_e1_i32_o16_se0.25".

#         Returns:
#             BlockArgs: namedtuple defined at the top of this function.
#         """
#         ops = block_string.split("_")
#         options = {}
#         for op in ops:
#             splits = re.split(r"(\d.*)", op)
#             if len(splits) >= 2:
#                 key, value = splits[:2]
#                 options[key] = value

#         # check stride
#         stride_check = (
#             ("s" in options and len(options["s"]) == 1)
#             or (len(options["s"]) == 2 and options["s"][0] == options["s"][1])
#             or (len(options["s"]) == 3 and options["s"][0] == options["s"][1] and options["s"][0] == options["s"][2])
#         )
#         if not stride_check:
#             raise ValueError("invalid stride option received")

#         return BlockArgs(
#             num_repeat=int(options["r"]),
#             kernel_size=int(options["k"]),
#             stride=int(options["s"][0]),
#             expand_ratio=int(options["e"]),
#             input_filters=int(options["i"]),
#             output_filters=int(options["o"]),
#             id_skip=("noskip" not in block_string),
#             se_ratio=float(options["se"]) if "se" in options else None,
#         )


#     def to_string(self):
#         """
#         Return a block string notation for current BlockArgs object

#         Returns:
#             A string notation of BlockArgs object arguments.
#                 Example: "r1_k3_s11_e1_i32_o16_se0.25_noskip".
#         """
#         string = "r{}_k{}_s{}{}_e{}_i{}_o{}_se{}".format(
#             self.num_repeat,
#             self.kernel_size,
#             self.stride,
#             self.stride,
#             self.expand_ratio,
#             self.input_filters,
#             self.output_filters,
#             self.se_ratio,
#         )

#         if not self.id_skip:
#             string += "_noskip"
#         return string
    
# class UpCat(nn.Module):
#     """upsampling, concatenation with the encoder feature map, two convolutions"""

#     @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
#     def __init__(
#         self,
#         spatial_dims: int,
#         in_chns: int,
#         cat_chns: int,
#         out_chns: int,
#         act: Union[str, tuple],
#         norm: Union[str, tuple],
#         bias: bool,
#         dropout: Union[float, tuple] = 0.0,
#         upsample: str = "deconv",
#         pre_conv: Optional[Union[nn.Module, str]] = "default",
#         interp_mode: str = "linear",
#         align_corners: Optional[bool] = True,
#         halves: bool = True,
#         dim: Optional[int] = None,
#         # is_pad: bool = True,
#         is_pad: bool = False,
#     ):
#         """
#         Args:
#             spatial_dims: number of spatial dimensions.
#             in_chns: number of input channels to be upsampled.
#             cat_chns: number of channels from the decoder.
#             out_chns: number of output channels.
#             act: activation type and arguments.
#             norm: feature normalization type and arguments.
#             bias: whether to have a bias term in convolution blocks.
#             dropout: dropout ratio. Defaults to no dropout.
#             upsample: upsampling mode, available options are
#                 ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
#             pre_conv: a conv block applied before upsampling.
#                 Only used in the "nontrainable" or "pixelshuffle" mode.
#             interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
#                 Only used in the "nontrainable" mode.
#             align_corners: set the align_corners parameter for upsample. Defaults to True.
#                 Only used in the "nontrainable" mode.
#             halves: whether to halve the number of channels during upsampling.
#                 This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.
#             is_pad: whether to pad upsampling features to fit features from encoder. Defaults to True.
#         .. deprecated:: 0.6.0
#             ``dim`` is deprecated, use ``spatial_dims`` instead.
#         """
#         super().__init__()
#         if dim is not None:
#             spatial_dims = dim
#         if upsample == "nontrainable" and pre_conv is None:
#             up_chns = in_chns
#         else:
#             up_chns = in_chns // 2 if halves else in_chns
#         self.upsample = UpSample(
#             spatial_dims,
#             in_chns,
#             up_chns,
#             2,
#             mode=upsample,
#             pre_conv=pre_conv,
#             interp_mode=interp_mode,
#             align_corners=align_corners,
#         )
#         self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)
#         self.is_pad = is_pad

#     def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
#         """
#         Args:
#             x: features to be upsampled.
#             x_e: features from the encoder.
#         """
#         x_0 = self.upsample(x)

#         if x_e is not None:
#             if self.is_pad:
#                 # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
#                 dimensions = len(x.shape) - 2
#                 sp = [0] * (dimensions * 2)
#                 for i in range(dimensions):
#                     if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
#                         sp[i * 2 + 1] = 1
#                 # x_0 = torch.nn.functional.pad(x_0, sp, "constant")
#                 x_0 = torch.nn.functional.pad(x_0, sp, "replicate") # original
#             x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
#         else:
#             x = self.convs(x_0)

#         return x

# backup
# class UNet(nn.Module):
#     def __init__(
#         self,
#         modelName='efficientnet-b4',
#         spatial_dims: int = 2,
#         in_channels: int = 1,
#         out_channels: int = 2,
#         act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
#         norm: Union[str, tuple] = ("instance", {"affine": True}),
#         bias: bool = True,
#         dropout: Union[float, tuple] = 0, # (0.1, {"inplace": True}),
#         upsample: str = "deconv", # [deconv, nontrainable, pixelshuffle]
#         nnblock = False, # [False, True]
#         ASPP = None, # [None, 'last', 'all']
#         supervision = None, #[None,'old','new']
#         FFC = None, # [None, 'FFC', 'DeepRFT']
#         acm = False, # [False, 'woLoss', 0<x<1]
#         TRB = False, # [False, True]
#         se_module= 'se'

#     ):
#         """
#         A UNet implementation with 1D/2D/3D supports.

#         Args:
#             modelName: Backbone of efficientNet
#             spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
#             in_channels: number of input channels. Defaults to 1.
#             out_channels: number of output channels. Defaults to 2.
#             act: activation type and arguments. Defaults to LeakyReLU.
#             norm: feature normalization type and arguments. Defaults to instance norm. for group norm, norm=("group", {"num_groups": 4})
#             bias: whether to have a bias term in convolution blocks. Defaults to True.
#                 According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
#                 if a conv layer is directly followed by a batch norm layer, bias should be False.
#             dropout: dropout ratio. Defaults to no dropout.
#             upsample: upsampling mode, available options are ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        
#         <example>
#         net = UNet(modelName='efficientnet-b2', spatial_dims = 1, in_channels = 1, out_channels = 4, norm='instance', upsample='pixelshuffle', nnblock=True, ASPP='all', supervision=True, FFC='FFC', TRB=True)
#         yhat = net(torch.rand(2,1,2048))

#         if isinstance(yhat, list) or isinstance(yhat, tuple):
#             for yhat_ in yhat:
#                 print(yhat_.shape)
#         else:
#             print(yhat.shape)
#         """
#         super().__init__()

#         # U-net encoder
#         if 'efficientnet' in modelName:
#             ########################################################## preset init_ch
#             # self.encoder = monai.networks.nets.EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=1, in_channels=1, norm=norm , num_classes=1000, adv_prop=True)
#             self.encoder =EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=1, in_channels=1, norm=norm , num_classes=1000, adv_prop=True,se_module=se_module)
#             x_test = torch.rand(2, 1, 2048)
#             yhat_test = self.encoder(x_test)
#             init_ch = yhat_test[0].shape[1]
#             ########################################################## preset init_ch
#             self.conv_0 = TwoConv(spatial_dims, in_channels, init_ch, act, norm, bias, dropout)
#             # self.encoder = monai.networks.nets.EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=1, in_channels=init_ch, norm=norm , num_classes=1000, adv_prop=True)
#             self.encoder = EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=1, in_channels=init_ch, norm=norm , num_classes=1000, adv_prop=True,se_module=se_module)
#         elif 'resnet' in modelName:
#             self.encoder = ResNetFeature(n_input_channels=1, block=ResNetBottleneck, layers= [3, 4, 6, 3], block_inplanes= [64, 128, 256, 512], spatial_dims=1, conv1_t_size=7, conv1_t_stride=2, no_max_pool= True)
#             x_test = torch.rand(2, 1, 2048)
#             yhat_test = self.encoder(x_test)
#             init_ch = yhat_test[0].shape[1]
#             ########################################################## preset init_ch
#             self.conv_0 = TwoConv(spatial_dims, in_channels, init_ch, act, norm, bias, dropout)
#             self.encoder = ResNetFeature(n_input_channels=init_ch, block=ResNetBottleneck, layers= [3, 4, 6, 3], block_inplanes= [64, 128, 256, 512], spatial_dims=1, conv1_t_size=7, conv1_t_stride=2, no_max_pool= True)
#         else:
#             print('please check modelName')
        
#         x = torch.rand(2, init_ch, 64)
#         yhat = self.encoder(x)
#         # fea = []
#         # for yhat_ in yhat:
#         #     fea.append(yhat_.shape[1])            
#         fea = [yhat_.shape[1] for yhat_ in yhat]
#         print(fea)
        
#         # bottleneck modules
#         self.acm = acm
#         if acm:
#             if acm =='woLoss':
#                 self.acm_1 = ACM(num_heads=fea[0]//8, num_features=fea[0], orthogonal_loss=False)
#                 self.acm_2 = ACM(num_heads=fea[1]//8, num_features=fea[1], orthogonal_loss=False)
#                 self.acm_3 = ACM(num_heads=fea[2]//8, num_features=fea[2], orthogonal_loss=False)
#                 self.acm_4 = ACM(num_heads=fea[3]//8, num_features=fea[3], orthogonal_loss=False)
#                 self.acm_5 = ACM(num_heads=fea[4]//8, num_features=fea[4], orthogonal_loss=False)
#             elif 0<acm<=1:
#                 self.acm_1 = ACM(num_heads=fea[0]//8, num_features=fea[0], orthogonal_loss=True)
#                 self.acm_2 = ACM(num_heads=fea[1]//8, num_features=fea[1], orthogonal_loss=True)
#                 self.acm_3 = ACM(num_heads=fea[2]//8, num_features=fea[2], orthogonal_loss=True)
#                 self.acm_4 = ACM(num_heads=fea[3]//8, num_features=fea[3], orthogonal_loss=True)
#                 self.acm_5 = ACM(num_heads=fea[4]//8, num_features=fea[4], orthogonal_loss=True)
                
#         self.nnblock = nnblock
#         if nnblock:
#             norms = ['instance','batch']
#             if not norm in norms:
#                 norm = None
#             self.nnblock1 = NLBlockND(in_channels=fea[0], mode='embedded', dimension=spatial_dims, norm_layer=norm)
#             self.nnblock2 = NLBlockND(in_channels=fea[1], mode='embedded', dimension=spatial_dims, norm_layer=norm)
#             self.nnblock3 = NLBlockND(in_channels=fea[2], mode='embedded', dimension=spatial_dims, norm_layer=norm)
#             self.nnblock4 = NLBlockND(in_channels=fea[3], mode='embedded', dimension=spatial_dims, norm_layer=norm)
#             self.nnblock5 = NLBlockND(in_channels=fea[4], mode='embedded', dimension=spatial_dims, norm_layer=norm)                      

#         self.FFC = FFC
#         if FFC=='FFC':
#             self.FFCblock1 = FFC_BN_ACT(fea[0],fea[0])
#             self.FFCblock2 = FFC_BN_ACT(fea[1],fea[1])
#             self.FFCblock3 = FFC_BN_ACT(fea[2],fea[2])
#             self.FFCblock4 = FFC_BN_ACT(fea[3],fea[3])
#             self.FFCblock5 = FFC_BN_ACT(fea[4],fea[4])            
#         elif FFC=='DeepRFT':
#             self.FFCblock1 = FFT_ConvBlock(fea[0],fea[0])
#             self.FFCblock2 = FFT_ConvBlock(fea[1],fea[1])
#             self.FFCblock3 = FFT_ConvBlock(fea[2],fea[2])
#             self.FFCblock4 = FFT_ConvBlock(fea[3],fea[3])
#             self.FFCblock5 = FFT_ConvBlock(fea[4],fea[4])
        
#         self.TRB = TRB
#         if TRB:
#             self.TRblock = monai.networks.blocks.TransformerBlock(hidden_size = 64, mlp_dim= 3072, num_heads=16, dropout_rate=0.1)
            
#         self.ASPP = ASPP        
#         if ASPP=='last':
#             self.ASPPblock = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims,
#                                                              in_channels=fea[4], 
#                                                              conv_out_channels=fea[4]//4,
#                                                              norm_type=norm,
#                                                              acti_type=act, 
#                                                              bias=bias)  
#         elif ASPP=='all':
#             self.ASPPblock1 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[0], conv_out_channels=fea[0]//4,
#                                                              norm_type=norm, acti_type=act, bias=bias)            
#             self.ASPPblock2 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[1], conv_out_channels=fea[1]//4,
#                                                              norm_type=norm, acti_type=act, bias=bias)            
#             self.ASPPblock3 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[2], conv_out_channels=fea[2]//4,
#                                                              norm_type=norm, acti_type=act, bias=bias)            
#             self.ASPPblock4 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[3], conv_out_channels=fea[3]//4,
#                                                              norm_type=norm, acti_type=act, bias=bias)            
#             self.ASPPblock5 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[4], conv_out_channels=fea[4]//4,
#                                                              norm_type=norm, acti_type=act, bias=bias)            
        
#         # U-Net Decoder
#         self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample, interp_mode='linear')
#         self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample, interp_mode='linear')
#         self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample, interp_mode='linear')
#         self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[0], act, norm, bias, dropout, upsample, interp_mode='linear')
#         self.upcat_0 = UpCat(spatial_dims, fea[0], fea[0], fea[0], act, norm, bias, dropout, upsample, interp_mode='linear', halves=False)
#         self.final_conv = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
        
#         self.supervision = supervision
#         if supervision=='old' or supervision =='type1':
#             self.final_conv = Conv["conv", spatial_dims](fea[0]+fea[0]+fea[1]+fea[2]+fea[3]+fea[4], out_channels, kernel_size=1)
#         elif supervision=='new' or supervision =='type2':
#             self.sv0= Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=3, padding=1)
#             self.sv1= Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=3, padding=1)
#             self.sv2= Conv["conv", spatial_dims](fea[1], out_channels, kernel_size=3, padding=1)
#             self.sv3= Conv["conv", spatial_dims](fea[2], out_channels, kernel_size=3, padding=1)
#             self.sv4= Conv["conv", spatial_dims](fea[3], out_channels, kernel_size=3, padding=1)
#             self.sv5= Conv["conv", spatial_dims](fea[4], out_channels, kernel_size=3, padding=1)
#             self.final_conv = Conv["conv", spatial_dims](out_channels*6, out_channels, kernel_size=1)

#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         set_seed()
#         if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
#             neg_slope=1e-2
#             module.weight = nn.init.kaiming_normal_(module.weight, a=neg_slope)
#             if module.bias is not None:
#                 module.bias = torch.nn.init.zeros_(module.bias)
#         if isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.BatchNorm1d):
#             if module.bias is not None:
#                 module.bias = torch.nn.init.zeros_(module.bias)

#     def forward(self, x: torch.Tensor):
        
#         x0 = self.conv_0(x)
#         x1, x2, x3, x4, x5 = self.encoder(x0)
#         # print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
#         dp = False
        
#         if self.acm=='woLoss':
#             x1 = x1 + self.acm_1(x1)
#             x2 = x2 + self.acm_2(x2)
#             x3 = x3 + self.acm_3(x3)
#             x4 = x4 + self.acm_4(x4)
#             x5 = x5 + self.acm_5(x5)
#         elif 0<self.acm<=1:
#             x1_, dp1 = self.acm_1(x1)
#             x2_, dp2 = self.acm_2(x2)
#             x3_, dp3 = self.acm_3(x3)
#             x4_, dp4 = self.acm_4(x4)
#             x5_, dp5 = self.acm_5(x5)
#             x1 = x1 + x1_
#             x2 = x2 + x2_
#             x3 = x3 + x3_
#             x4 = x4 + x4_
#             x5 = x5 + x5_
#             dp = dp1+dp2+dp3+dp4+dp5
#             dp = torch.abs(torch.mean(dp))*self.acm
#         else:
#             pass
            
#         if self.FFC:
#             x1 = x1 + self.FFCblock1(x1)
#             x2 = x2 + self.FFCblock2(x2)
#             x3 = x3 + self.FFCblock3(x3)
#             x4 = x4 + self.FFCblock4(x4)
#             x5 = x5 + self.FFCblock5(x5)

#         if self.nnblock:
#             x1 = x1 + self.nnblock1(x1)
#             x2 = x2 + self.nnblock2(x2)
#             x3 = x3 + self.nnblock3(x3)
#             x4 = x4 + self.nnblock4(x4)
#             x5 = x5 + self.nnblock5(x5)

#         if self.TRB:
#             x5 = x5 + self.TRblock(x5)
            
#         if self.ASPP=='last':
#             x5 = x5 + self.ASPPblock(x5)            
            
#         elif self.ASPP=='all': 
#             x1 = x1 + self.ASPPblock1(x1)
#             x2 = x2 + self.ASPPblock2(x2)
#             x3 = x3 + self.ASPPblock3(x3)
#             x4 = x4 + self.ASPPblock4(x4)
#             x5 = x5 + self.ASPPblock5(x5)
            
#         u4 = self.upcat_4(x5, x4)
#         u3 = self.upcat_3(u4, x3)
#         u2 = self.upcat_2(u3, x2)
#         u1 = self.upcat_1(u2, x1)
#         u0 = self.upcat_0(u1, x0)
#         # print(u0.shape, u1.shape, u2.shape, u3.shape, u4.shape)
        
#         if self.supervision=='old' or self.supervision =='type1':            
#             s5 = _upsample_like(x5,u0)
#             s4 = _upsample_like(x4,u0)
#             s3 = _upsample_like(x3,u0)
#             s2 = _upsample_like(x2,u0)
#             s1 = _upsample_like(x1,u0)            
#             u0 = torch.cat((u0,s1,s2,s3,s4,s5),1)            
#             # print(u0.shape, s1.shape, s2.shape, s3.shape, s4.shape, s5.shape)
            
#             logits = self.final_conv(u0)
#             # print(logits.shape)
#             # return torch.sigmoid(logits)
#             return [torch.sigmoid(logits), dp] if dp else torch.sigmoid(logits)

#         elif self.supervision=='new' or self.supervision =='type2':
#             s5 = self.sv5(_upsample_like(x5,u0))
#             s4 = self.sv4(_upsample_like(x4,u0))
#             s3 = self.sv3(_upsample_like(x3,u0))
#             s2 = self.sv2(_upsample_like(x2,u0))
#             s1 = self.sv1(_upsample_like(x1,u0))
#             u0 = self.sv0(u0)
#             # print(u0.shape, s1.shape, s2.shape, s3.shape, s4.shape, s5.shape)
#             u0 = torch.cat((u0,s1,s2,s3,s4,s5),dim=1)
            
#             logits = self.final_conv(u0)
#             # print(logits.shape)
#             # return torch.sigmoid(logits), torch.sigmoid(s1), torch.sigmoid(s2), torch.sigmoid(s3), torch.sigmoid(s4), torch.sigmoid(s5) 
#             return [torch.sigmoid(logits), dp] if dp else torch.sigmoid(logits)
        
#         else:
#             logits = self.final_conv(u0)
#             # print(logits.shape)
#             return [torch.sigmoid(logits), dp] if dp else torch.sigmoid(logits)

# # featureLength = 1024
# featureLength = 1280

# class UNet(nn.Module):
#     def __init__(
#         self,
#         modelName='efficientnet-b0',
#         spatial_dims: int = 2,
#         in_channels: int = 1,
#         out_channels: int = 2,
#         act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
#         norm: Union[str, tuple] = ("instance", {"affine": True}),
#         bias: bool = True,
#         dropout: Union[float, tuple] = 0, # (0.1, {"inplace": True}),
#         upsample: str = "deconv", # [deconv, nontrainable, pixelshuffle]
#         supervision = "NONE", #[None,'old','new']
#         encModule="NONE",
#         decModule="NONE",
#         segheadModule ="NONE",
#         # se_module= 'se',
#         mtl="NONE"
#     ):
#         """
#         A UNet implementation with 1D/2D/3D supports.

#         Args:
#             modelName: Backbone of efficientNet
#             spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
#             in_channels: number of input channels. Defaults to 1.
#             out_channels: number of output channels. Defaults to 2.
#             act: activation type and arguments. Defaults to LeakyReLU.
#             norm: feature normalization type and arguments. Defaults to instance norm. for group norm, norm=("group", {"num_groups": 4})
#             bias: whether to have a bias term in convolution blocks. Defaults to True.
#                 According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
#                 if a conv layer is directly followed by a batch norm layer, bias should be False.
#             dropout: dropout ratio. Defaults to no dropout.
#             upsample: upsampling mode, available options are ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
#             supervision : 'TYPE1,'TYPE2,'NONE'
#             se_module : 'se', 'acm', 'ffc', 'deeprft', 'nlnn'
#             encModule : '[Module]_[BOTTOM#]'-->'FFC','DEEPRFT','ACM8','NONE','ACM2','ACM4','NLNN','SE'
#                                              -->'BOTTOM5','BOTTOM4','BOTTOM3','BOTTOM2','BOTTOM1'
            
#         <example>
#         net = UNet(modelName='efficientnet-b2', spatial_dims = 1, in_channels = 1, out_channels = 4, norm='instance', upsample='pixelshuffle', nnblock=True, ASPP='all', supervision=True, FFC='FFC', TRB=True)
#         yhat = net(torch.rand(2,1,2048))

#         """
#         super().__init__()

#         # U-net encoder
#         if 'efficientnet' in modelName:
#             ########################################################## preset init_ch
#             self.encoder = monai.networks.nets.EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=spatial_dims, in_channels=1, norm=norm , num_classes=1000, adv_prop=True)
#             # self.encoder =EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=1, in_channels=1, norm=norm , num_classes=1000, adv_prop=True,se_module=se_module)
#             x_test = torch.rand(2, 1, 2048)
#             yhat_test = self.encoder(x_test)
#             init_ch = yhat_test[0].shape[1]
#             ########################################################## preset init_ch
#             self.conv_0 = TwoConv(spatial_dims, in_channels, init_ch, act, norm, bias, dropout)
#             self.encoder = monai.networks.nets.EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=spatial_dims, in_channels=init_ch, norm=norm , num_classes=1000, adv_prop=True)
#             # self.encoder = EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=1, in_channels=init_ch, norm=norm , num_classes=1000, adv_prop=True,se_module=se_module)
#         elif 'resnet' in modelName:
#             if modelName == 'resnet18':
#                 self.encoder = resnet18(spatial_dims=spatial_dims, n_input_channels=in_channels)
#             elif modelName == 'resnet34':
#                 self.encoder = resnet34(spatial_dims=spatial_dims, n_input_channels=in_channels)
#             elif modelName == 'resnet50':
#                 self.encoder = resnet50(spatial_dims=spatial_dims, n_input_channels=in_channels)

#             x_test = torch.rand(2, in_channels, 1280)
#             yhat_test = self.encoder(x_test)
#             init_ch = yhat_test[0].shape[1]
#             ########################################################## preset init_ch
#             self.conv_0 = TwoConv(spatial_dims, in_channels, init_ch, act, norm, bias, dropout)
#             if modelName == 'resnet18':
#                 self.encoder = resnet18(spatial_dims=spatial_dims, n_input_channels=init_ch)
#             elif modelName == 'resnet34':
#                 self.encoder = resnet34(spatial_dims=spatial_dims, n_input_channels=init_ch)
#             elif modelName == 'resnet50':
#                 self.encoder = resnet50(spatial_dims=spatial_dims, n_input_channels=init_ch)
#             self.encoder = bn2instance(self.encoder)
#         else:
#             print('please check modelName')
        
#         x = torch.rand(2, init_ch, 64)
#         yhat = self.encoder(x)
#         fea = [yhat_.shape[1] for yhat_ in yhat]
#         print(fea)
        
#         # skip modules
#         self.encModule = encModule
#         self.encModule1 = nn.Identity()
#         self.encModule2 = nn.Identity()
#         self.encModule3 = nn.Identity()
#         self.encModule4 = nn.Identity()
#         self.encModule5 = nn.Identity()
#         print(f"U-NET encModule is {encModule}")

#         if 'ACM' in encModule:
#             group = 4
#             self.ACMLambda = 0.01
#             print(f"encModule: {encModule} group {group} ACMLambda {self.ACMLambda}")
#             if self.ACMLambda==0:
#                 self.encModule1 = ACM(num_heads=fea[0]//group, num_features=fea[0], orthogonal_loss=False)# if l>=5 else nn.Identity()
#                 self.encModule2 = ACM(num_heads=fea[1]//group, num_features=fea[1], orthogonal_loss=False)# if l>=4 else nn.Identity()
#                 self.encModule3 = ACM(num_heads=fea[2]//group, num_features=fea[2], orthogonal_loss=False)# if l>=3 else nn.Identity()
#                 self.encModule4 = ACM(num_heads=fea[3]//group, num_features=fea[3], orthogonal_loss=False)# if l>=2 else nn.Identity()
#                 self.encModule5 = ACM(num_heads=fea[4]//group, num_features=fea[4], orthogonal_loss=False)# if l>=1 else nn.Identity()
#             # else:
#             #     self.encModule1 = ACM(num_heads=fea[0]//group, num_features=fea[0], orthogonal_loss=True)# if l>=5 else nn.Identity()
#             #     self.encModule2 = ACM(num_heads=fea[1]//group, num_features=fea[1], orthogonal_loss=True)# if l>=4 else nn.Identity()
#             #     self.encModule3 = ACM(num_heads=fea[2]//group, num_features=fea[2], orthogonal_loss=True)# if l>=3 else nn.Identity()
#             #     self.encModule4 = ACM(num_heads=fea[3]//group, num_features=fea[3], orthogonal_loss=True)# if l>=2 else nn.Identity()
#             #     self.encModule5 = ACM(num_heads=fea[4]//group, num_features=fea[4], orthogonal_loss=True)# if l>=1 else nn.Identity()
#         elif 'NLNN' in encModule:
#             norms = ['instance','batch']
#             if not norm in norms:
#                 norm = None
#             # l = int(encModule.split('BOTTOM')[-1])
#             # print(f"encModule: {encModule} {l}")
#             self.encModule1 = NLBlockND(in_channels=fea[0], mode='embedded', dimension=spatial_dims, norm_layer=norm)# if l>=5 else nn.Identity()
#             self.encModule2 = NLBlockND(in_channels=fea[1], mode='embedded', dimension=spatial_dims, norm_layer=norm)# if l>=4 else nn.Identity()
#             self.encModule3 = NLBlockND(in_channels=fea[2], mode='embedded', dimension=spatial_dims, norm_layer=norm)# if l>=3 else nn.Identity()
#             self.encModule4 = NLBlockND(in_channels=fea[3], mode='embedded', dimension=spatial_dims, norm_layer=norm)# if l>=2 else nn.Identity()
#             self.encModule5 = NLBlockND(in_channels=fea[4], mode='embedded', dimension=spatial_dims, norm_layer=norm)# if l>=1 else nn.Identity()
#         elif 'FFC' in encModule:
#             # l = int(encModule.split('BOTTOM')[-1])
#             # print(f"encModule: {encModule}")
#             self.encModule1 = FFC_BN_ACT(fea[0],fea[0])# if l>=5 else nn.Identity()
#             self.encModule2 = FFC_BN_ACT(fea[1],fea[1])# if l>=4 else nn.Identity()
#             self.encModule3 = FFC_BN_ACT(fea[2],fea[2])# if l>=3 else nn.Identity()
#             self.encModule4 = FFC_BN_ACT(fea[3],fea[3])# if l>=2 else nn.Identity()
#             self.encModule5 = FFC_BN_ACT(fea[4],fea[4])# if l>=1 else nn.Identity()
#         elif 'DEEPRFT' in encModule:
#             # l = int(encModule.split('BOTTOM')[-1])
#             # print(f"encModule: {encModule}")
#             self.encModule1 = FFT_ConvBlock(fea[0],fea[0])# if l>=5 else nn.Identity()
#             self.encModule2 = FFT_ConvBlock(fea[1],fea[1])# if l>=4 else nn.Identity()
#             self.encModule3 = FFT_ConvBlock(fea[2],fea[2])# if l>=3 else nn.Identity()
#             self.encModule4 = FFT_ConvBlock(fea[3],fea[3])# if l>=2 else nn.Identity()
#             self.encModule5 = FFT_ConvBlock(fea[4],fea[4])# if l>=1 else nn.Identity()
#         elif 'SE' in encModule:
#             # l = int(encModule.split('BOTTOM')[-1])
#             # print(f"encModule: {encModule}")
#             self.encModule1 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[0])# if l>=5 else nn.Identity()
#             self.encModule2 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[1])# if l>=4 else nn.Identity()
#             self.encModule3 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[2])# if l>=3 else nn.Identity()
#             self.encModule4 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[3])# if l>=2 else nn.Identity()
#             self.encModule5 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[4])# if l>=1 else nn.Identity()
#         elif 'CBAM' in encModule:
#             # l = int(encModule.split('BOTTOM')[-1])
#             # print(f"encModule: {encModule}")
#             self.encModule1 = CBAM(gate_channels=fea[0], reduction_ratio=16, pool_types=['avg', 'max'])# if l>=5 else nn.Identity()
#             self.encModule2 = CBAM(gate_channels=fea[1], reduction_ratio=16, pool_types=['avg', 'max'])# if l>=4 else nn.Identity()
#             self.encModule3 = CBAM(gate_channels=fea[2], reduction_ratio=16, pool_types=['avg', 'max'])# if l>=3 else nn.Identity()
#             self.encModule4 = CBAM(gate_channels=fea[3], reduction_ratio=16, pool_types=['avg', 'max'])# if l>=2 else nn.Identity()
#             self.encModule5 = CBAM(gate_channels=fea[4], reduction_ratio=16, pool_types=['avg', 'max'])# if l>=1 else nn.Identity()
#         elif 'MHA' in encModule:
#             # l = int(encModule.split('BOTTOM')[-1])
#             # print(f"encModule: {encModule}")
#             self.encModule1 = nn.MultiheadAttention(featureLength//2, 8, batch_first=True, dropout=0.01)# if l>=5 else nn.Identity()
#             self.encModule2 = nn.MultiheadAttention(featureLength//4, 8, batch_first=True, dropout=0.01)# if l>=4 else nn.Identity()
#             self.encModule3 = nn.MultiheadAttention(featureLength//8, 8, batch_first=True, dropout=0.01)# if l>=3 else nn.Identity()
#             self.encModule4 = nn.MultiheadAttention(featureLength//16, 8, batch_first=True, dropout=0.01)# if l>=2 else nn.Identity()
#             self.encModule5 = nn.MultiheadAttention(featureLength//32, 8, batch_first=True, dropout=0.01)# if l>=1 else nn.Identity()
            
#         # multiTaskCLS
#         self.mtl = mtl   
#         self.mtl_cls = nn.Identity()     
#         self.mtl_rec = nn.Identity()     
#         print(f"U-NET mtl is {mtl}")

#         if mtl == "CLS" or mtl=="ALL":
#             self.mtl_cls = nn.Sequential(monai.networks.blocks.ResidualSELayer(spatial_dims, fea[4]),nn.AdaptiveAvgPool1d(1),nn.Conv1d(fea[4],1,1))

#         if mtl == "REC" or mtl=="ALL":
#             mtl4 = Convolution(spatial_dims=spatial_dims, in_channels=fea[4], out_channels=fea[3], adn_ordering="ADN", act=("prelu", {"init": 0.2}), dropout=0.1, norm=norm)
#             mtl3 = Convolution(spatial_dims=spatial_dims, in_channels=fea[3], out_channels=fea[2], adn_ordering="ADN", act=("prelu", {"init": 0.2}), dropout=0.1, norm=norm)
#             mtl2 = Convolution(spatial_dims=spatial_dims, in_channels=fea[2], out_channels=fea[1], adn_ordering="ADN", act=("prelu", {"init": 0.2}), dropout=0.1, norm=norm)
#             mtl1 = Convolution(spatial_dims=spatial_dims, in_channels=fea[1], out_channels=fea[0], adn_ordering="ADN", act=("prelu", {"init": 0.2}), dropout=0.1, norm=norm)
#             mtl0 = Convolution(spatial_dims=spatial_dims, in_channels=fea[0], out_channels=in_channels, adn_ordering="ADN", act=("prelu", {"init": 0.2}), dropout=0.1, norm=norm)
#             self.mtl_rec = nn.Sequential(mtl4,nn.Upsample(scale_factor=2),mtl3,nn.Upsample(scale_factor=2),mtl2,nn.Upsample(scale_factor=2),mtl1,nn.Upsample(scale_factor=2),mtl0,nn.Upsample(scale_factor=2))
        
#         # U-Net Decoder
#         self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample, interp_mode='linear')
#         self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample, interp_mode='linear')
#         self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample, interp_mode='linear')
#         self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[0], act, norm, bias, dropout, upsample, interp_mode='linear')
#         self.upcat_0 = UpCat(spatial_dims, fea[0], fea[0], fea[0], act, norm, bias, dropout, upsample, interp_mode='linear', halves=False)

#         self.decModule =  decModule
#         print('U-NET decModule is', decModule)
#         self.decModule1 = nn.Identity()
#         self.decModule2 = nn.Identity()
#         self.decModule3 = nn.Identity()
#         self.decModule4 = nn.Identity()

#         if 'SE' in self.decModule:
#             self.decModule1 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[0]) # (128x256 and 512x256)
#             self.decModule2 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[1])
#             self.decModule3 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[2])
#             self.decModule4 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[3])   

#         elif 'NN' in self.decModule:
#             self.decModule1 = NLBlockND(in_channels=fea[0], mode='embedded', dimension=spatial_dims, norm_layer=norm)
#             self.decModule2 = NLBlockND(in_channels=fea[1], mode='embedded', dimension=spatial_dims, norm_layer=norm)
#             self.decModule3 = NLBlockND(in_channels=fea[2], mode='embedded', dimension=spatial_dims, norm_layer=norm)
#             self.decModule4 = NLBlockND(in_channels=fea[3], mode='embedded', dimension=spatial_dims, norm_layer=norm)               

#         elif 'FFC' in self.decModule:
#             self.decModule1 = FFC_BN_ACT(fea[0],fea[0])
#             self.decModule2 = FFC_BN_ACT(fea[1],fea[1])
#             self.decModule3 = FFC_BN_ACT(fea[2],fea[2])
#             self.decModule4 = FFC_BN_ACT(fea[3],fea[3])

#         elif 'DEEPRFT' in self.decModule:
#             self.decModule1 = FFT_ConvBlock(fea[0],fea[0])
#             self.decModule2 = FFT_ConvBlock(fea[1],fea[1])
#             self.decModule3 = FFT_ConvBlock(fea[2],fea[2])
#             self.decModule4 = FFT_ConvBlock(fea[3],fea[3])
            
#         elif 'ACM' in self.decModule:
#             # group = 4
#             # self.decModule1 = ACM(num_heads=fea[0]//group, num_features=fea[0], orthogonal_loss=False)
#             # self.decModule2 = ACM(num_heads=fea[1]//group, num_features=fea[1], orthogonal_loss=False)
#             # self.decModule3 = ACM(num_heads=fea[2]//group, num_features=fea[2], orthogonal_loss=False)
#             # self.decModule4 = ACM(num_heads=fea[3]//group, num_features=fea[3], orthogonal_loss=False)
#             # self.decModule5 = ACM(num_heads=fea[4]//group, num_features=fea[4], orthogonal_loss=False)
#             # self.decModule6 = ACM(num_heads=fea[5]//group, num_features=fea[5], orthogonal_loss=False)
#             group = 4
#             self.decModule1 = ACM(num_heads=fea[0]//group, num_features=fea[0], orthogonal_loss=False)
#             self.decModule2 = ACM(num_heads=fea[1]//group, num_features=fea[1], orthogonal_loss=False)
#             self.decModule3 = ACM(num_heads=fea[2]//group, num_features=fea[2], orthogonal_loss=False)
#             self.decModule4 = ACM(num_heads=fea[3]//group, num_features=fea[3], orthogonal_loss=False)

#         elif 'MHA' in self.decModule:
#             featureLength = 1280
#             self.decModule1 = nn.MultiheadAttention(featureLength//1, 8, batch_first=True, dropout=0.01) 
#             self.decModule2 = nn.MultiheadAttention(featureLength//2, 8, batch_first=True, dropout=0.01)
#             self.decModule3 = nn.MultiheadAttention(featureLength//4, 8, batch_first=True, dropout=0.01)
#             self.decModule4 = nn.MultiheadAttention(featureLength//8, 8, batch_first=True, dropout=0.01)

#         self.supervision = supervision
#         if supervision == 'NONE':
#             supervision_c = fea[0]
#         elif supervision =='TYPE1':
#             supervision_c = fea[0]+fea[0]+fea[1]+fea[2]+fea[3]+fea[4]
#         elif supervision =='TYPE2':
#             self.sv0= Conv["conv", spatial_dims](fea[0], out_channels*8, kernel_size=3, padding=1)
#             self.sv1= Conv["conv", spatial_dims](fea[0], out_channels*8, kernel_size=3, padding=1)
#             self.sv2= Conv["conv", spatial_dims](fea[1], out_channels*8, kernel_size=3, padding=1)
#             self.sv3= Conv["conv", spatial_dims](fea[2], out_channels*8, kernel_size=3, padding=1)
#             self.sv4= Conv["conv", spatial_dims](fea[3], out_channels*8, kernel_size=3, padding=1)
#             self.sv5= Conv["conv", spatial_dims](fea[4], out_channels*8, kernel_size=3, padding=1)
#             supervision_c =  out_channels*8*6
            
#         self.segheadModule = segheadModule
#         print(f"U-NET segheadModule is {segheadModule}")

#         self.segheadModule = nn.Identity()

#         if 'ACM' in segheadModule:
#             group = int(segheadModule.split('ACM')[-1][0])
#             self.ACMLambda = 0.01
#             print(f"segheadModule: {segheadModule} group {group} ACMLambda {self.ACMLambda}")
#             if self.ACMLambda==0:
#                 self.segheadModule = ACM(num_heads=supervision_c//group, num_features=supervision_c, orthogonal_loss=False)
#             else:
#                 self.segheadModule = ACM(num_heads=supervision_c//group, num_features=supervision_c, orthogonal_loss=True)

#         elif 'NLNN' in segheadModule:
#             norms = ['instance','batch']
#             if not norm in norms:
#                 norm = None
#             self.segheadModule = NLBlockND(in_channels=supervision_c, mode='embedded', dimension=spatial_dims, norm_layer=norm)

#         elif 'FFC' in segheadModule:
#             self.segheadModule = FFC_BN_ACT(supervision_c,supervision_c)       

#         elif 'DEEPRFT' in segheadModule:
#             self.segheadModule = FFT_ConvBlock(supervision_c,supervision_c)

#         elif 'SE' in segheadModule:
#             self.segheadModule = monai.networks.blocks.ResidualSELayer(spatial_dims, supervision_c)
            
#         elif 'CBAM' in segheadModule:
#             self.segheadModule = CBAM(gate_channels=supervision_c, reduction_ratio=16, pool_types=['avg', 'max'])

#         self.final_conv = nn.Sequential(self.segheadModule, Conv["conv", spatial_dims](supervision_c, out_channels, kernel_size=1),)
                                    
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         # set_seed()
#         if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
#             neg_slope=1e-2
#             module.weight = nn.init.kaiming_normal_(module.weight, a=neg_slope)
#             if module.bias is not None:
#                 module.bias = torch.nn.init.zeros_(module.bias)
#         if isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.BatchNorm1d):
#             if module.bias is not None:
#                 module.bias = torch.nn.init.zeros_(module.bias)

#     def forward(self, x: torch.Tensor):
        
#         x0 = self.conv_0(x)
#         x1, x2, x3, x4, x5 = self.encoder(x0)
#         dp = False
#         dp1 = dp2 = dp3 = dp4 = dp5 = 0.
        
#         if self.encModule=="NONE":
#             pass
#         elif "MHA" in self.encModule:
#             x1,_ = self.encModule1(x1,x1,x1)
#             x2,_ = self.encModule2(x2,x2,x2)
#             x3,_ = self.encModule3(x3,x3,x3)
#             x4,_ = self.encModule4(x4,x4,x4)
#             x5,_ = self.encModule5(x5,x5,x5)
#         else:
#             x1 = self.encModule1(x1)
#             x2 = self.encModule2(x2)
#             x3 = self.encModule3(x3)
#             x4 = self.encModule4(x4)
#             x5 = self.encModule5(x5)
        
# #             x1 = x1 + self.encModule1(x1)
# #             x2 = x2 + self.encModule2(x2)
# #             x3 = x3 + self.encModule3(x3)
# #             x4 = x4 + self.encModule4(x4)
# #             x5 = x5 + self.encModule5(x5)
            
#             if isinstance(x1,tuple) or isinstance(x1,list):
#                 x1, dp1 = x1
#                 dp1 = torch.abs(dp1.mean())
#             if isinstance(x2,tuple) or isinstance(x2,list):
#                 x2, dp2 = x2 
#                 dp2 = torch.abs(dp2.mean())
#             if isinstance(x3,tuple) or isinstance(x3,list):
#                 x3, dp3 = x3 
#                 dp3 = torch.abs(dp3.mean())
#             if isinstance(x4,tuple) or isinstance(x4,list):
#                 x4, dp4 = x4 
#                 dp4 = torch.abs(dp4.mean())
#             if isinstance(x5,tuple) or isinstance(x5,list):
#                 x5, dp5 = x5
#                 dp5 = torch.abs(dp5.mean())
#             if dp1!=0 or dp2!=0 or dp!=0 or dp4!=0 or dp5!=0:
#                 dp = self.ACMLambda * (dp1+dp2+dp3+dp4+dp5)
                                        
#         out_cls = None    
#         out_rec = None        
#         if self.mtl_cls:
#             out_cls = self.mtl_cls(x5)
#         if self.mtl_rec:
#             out_rec = self.mtl_rec(x5)
  
#         u4 = self.upcat_4(x5, x4)
#         u4 = self.decModule4(u4)
#         u3 = self.upcat_3(u4, x3)
#         u3 = self.decModule3(u3)
#         u2 = self.upcat_2(u3, x2)
#         u2 = self.decModule2(u2)
#         u1 = self.upcat_1(u2, x1)
#         u1 = self.decModule1(u1)
#         u0 = self.upcat_0(u1, x0)
#         # print(u0.shape, u1.shape, u2.shape, u3.shape, u4.shape)
        
#         if self.supervision=='old' or self.supervision =='TYPE1':            
#             s5 = _upsample_like(x5,u0)
#             s4 = _upsample_like(x4,u0)
#             s3 = _upsample_like(x3,u0)
#             s2 = _upsample_like(x2,u0)
#             s1 = _upsample_like(x1,u0)            
#             u0 = torch.cat((u0,s1,s2,s3,s4,s5),dim=1)            
#             # print(u0.shape, s1.shape, s2.shape, s3.shape, s4.shape, s5.shape)
            
#             logits = self.final_conv(u0)
#             # print(logits.shape)
                                        
#             # return [torch.sigmoid(logits), dp] if dp else torch.sigmoid(logits) # ACM lambda
#             if self.mtl == None:
#                 return torch.sigmoid(logits)
#             elif self.mtl == "ALL":
#                 return [torch.sigmoid(logits), torch.sigmoid(out_cls), torch.tanh(out_rec)]
#             elif self.mtl == "CLS":
#                 return [torch.sigmoid(logits), torch.sigmoid(out_cls)]
#             elif self.mtl == "REC":
#                 return [torch.sigmoid(logits), torch.tanh(out_rec)]

#         elif self.supervision=='new' or self.supervision =='TYPE2':
#             s5 = self.sv5(_upsample_like(x5,u0))
#             s4 = self.sv4(_upsample_like(x4,u0))
#             s3 = self.sv3(_upsample_like(x3,u0))
#             s2 = self.sv2(_upsample_like(x2,u0))
#             s1 = self.sv1(_upsample_like(x1,u0))
#             u0 = self.sv0(u0)
#             # print(u0.shape, s1.shape, s2.shape, s3.shape, s4.shape, s5.shape)
#             u0 = torch.cat((u0,s1,s2,s3,s4,s5),dim=1)
            
#             logits = self.final_conv(u0)
#             # print(logits.shape)
#             # return torch.sigmoid(logits), torch.sigmoid(s1), torch.sigmoid(s2), torch.sigmoid(s3), torch.sigmoid(s4), torch.sigmoid(s5) 
#             # return [torch.sigmoid(logits), dp] if dp else torch.sigmoid(logits)

#             if self.mtl == None:
#                 return torch.sigmoid(logits)
#             elif self.mtl == "ALL":
#                 return [torch.sigmoid(logits), torch.sigmoid(out_cls), torch.tanh(out_rec)]
#             elif self.mtl == "CLS":
#                 return [torch.sigmoid(logits), torch.sigmoid(out_cls)]
#             elif self.mtl == "REC":
#                 return [torch.sigmoid(logits), torch.tanh(out_rec)]
        
#         else:
#             logits = self.final_conv(u0)
#             # print(logits.shape)
#             # return [torch.sigmoid(logits), dp] if dp else torch.sigmoid(logits)
#             if self.mtl == None:
#                 return torch.sigmoid(logits)
#             elif self.mtl == "ALL":
#                 return [torch.sigmoid(logits), torch.sigmoid(out_cls), torch.tanh(out_rec)]
#             elif self.mtl == "CLS":
#                 return [torch.sigmoid(logits), torch.sigmoid(out_cls)]
#             elif self.mtl == "REC":
#                 return [torch.sigmoid(logits), torch.tanh(out_rec)]