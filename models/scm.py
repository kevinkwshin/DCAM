import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Act, Conv, Norm, Pool, split_args

# class ChannelSELayer(nn.Module):
#     """
#     Re-implementation of the Squeeze-and-Excitation block based on:
#     "Hu et al., Squeeze-and-Excitation Networks, https://arxiv.org/abs/1709.01507".
#     """

#     def __init__(
#         self,
#         spatial_dims: int,
#         in_channels: int,
#         r: int = 2,
#         acti_type_1: Union[Tuple[str, Dict], str] = ("relu", {"inplace": True}),
#         acti_type_2: Union[Tuple[str, Dict], str] = "sigmoid",
#         add_residual: bool = False,
#         beforeSE = None
#     ) -> None:
#         """
#         Args:
#             spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
#             in_channels: number of input channels.
#             r: the reduction ratio r in the paper. Defaults to 2.
#             acti_type_1: activation type of the hidden squeeze layer. Defaults to ``("relu", {"inplace": True})``.
#             acti_type_2: activation type of the output squeeze layer. Defaults to "sigmoid".

#         Raises:
#             ValueError: When ``r`` is nonpositive or larger than ``in_channels``.

#         See also:

#             :py:class:`monai.networks.layers.Act`

#         """
#         super().__init__()

#         self.add_residual = add_residual

#         pool_type = Pool[Pool.ADAPTIVEAVG, spatial_dims]
#         self.avg_pool = pool_type(1)  # spatial size (1, 1, ...)

#         channels = int(in_channels // r)
#         if channels <= 0:
#             raise ValueError(f"r must be positive and smaller than in_channels, got r={r} in_channels={in_channels}.")

#         act_1, act_1_args = split_args(acti_type_1)
#         act_2, act_2_args = split_args(acti_type_2)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, channels, bias=True),
#             Act[act_1](**act_1_args),
#             nn.Linear(channels, in_channels, bias=True),
#             Act[act_2](**act_2_args),
#         )
        
#         if beforeSE == None:
#             self.beforeSE = nn.Identity()
#         else:
#             self.beforeSE = beforeSE
        

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: in shape (batch, in_channels, spatial_1[, spatial_2, ...]).
#         """
#         b, c = x.shape[:2]
#         y: torch.Tensor = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view([b, c] + [1] * (x.ndim - 2))
        
#         x = self.beforeSE(x) # added
#         result = x * y

#         if self.add_residual:
#             result += x

#         return result



# class ResidualSELayer(ChannelSELayer):
#     """
#     A "squeeze-and-excitation"-like layer with a residual connection::

#         --+-- SE --o--
#           |        |
#           +--------+
#     """

#     def __init__(
#         self,
#         spatial_dims: int,
#         in_channels: int,
#         r: int = 2,
#         acti_type_1: Union[Tuple[str, Dict], str] = "leakyrelu",
#         acti_type_2: Union[Tuple[str, Dict], str] = "relu",
#     ) -> None:
#         """
#         Args:
#             spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
#             in_channels: number of input channels.
#             r: the reduction ratio r in the paper. Defaults to 2.
#             acti_type_1: defaults to "leakyrelu".
#             acti_type_2: defaults to "relu".

#         See also:
#             :py:class:`monai.networks.blocks.ChannelSELayer`
#         """
#         super().__init__(
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             r=r,
#             acti_type_1=acti_type_1,
#             acti_type_2=acti_type_2,
#             add_residual=True,
#         )

class SCM(nn.Module):
    """
    if __name__ == '__main__':
        import torch
        x1 = torch.randn(256 * 20 * 20 * 5).view(5, 256, 20, 20).float()
        x1 = torch.rand(2, 320, 160).float()
        scm = SCM(num_heads=32, num_features=320)
        scm.init_parameters()
        y, dp = scm(x1)
        print(y.shape)
        print(dp.shape)

    """
    def __init__(self, num_heads, num_features, scm_type=4, se= False):
        super(SCM, self).__init__()

        assert num_features % num_heads == 0

        self.num_features = num_features
        self.num_heads = num_heads

        self.add_mod = AttendModule(self.num_features, num_heads=num_heads)
        self.sub_mod = AttendModule(self.num_features, num_heads=num_heads)
        self.mul_mod = ModulateModule(channel=self.num_features, num_groups=num_heads, compressions=2)
        
        self.conv  = nn.Conv1d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.fft_conv  = nn.Conv1d(num_features*2, num_features*2, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.InstanceNorm1d(num_features)
        self.norm2 = nn.InstanceNorm1d(num_features*2)
 
        # self.se = monai.networks.blocks.ResidualSELayer(1, num_heads)
        
        # se module    
        if se!=False:
            spatial_dims = 1
            r = 2
            pool_type = Pool[Pool.ADAPTIVEAVG, spatial_dims]
            self.avg_pool = pool_type(1)  # spatial size (1, 1, ...)

            channels = int(num_features // r)
            if channels <= 0:
                raise ValueError(f"r must be positive and smaller than in_channels, got r={r} in_channels={num_features}.")

            self.se = nn.Sequential(
                nn.Linear(num_features, channels, bias=True),nn.GELU(),
                nn.Linear(channels, num_features, bias=True),nn.GELU(),
            )
        else:
            self.se = False

        self.scm_type = scm_type
        self.init_parameters()

    def init_parameters(self):
        if self.add_mod is not None:
            self.add_mod.init_parameters()
        if self.sub_mod is not None:
            self.sub_mod.init_parameters()
        if self.mul_mod is not None:
            self.mul_mod.init_parameters()

    def forward(self, x):
        
        # x = self.norm1(x)
        mu = x.mean([2], keepdim=True)
        x_mu = x - mu
        # print('mu',mu.shape, 'x_mu', x_mu.shape)

        # creates multipying feature
        mul_feature = self.mul_mod(mu)  # P

        # creates add or sub feature
        add_feature = self.add_mod(x_mu)  # K
        sub_feature = self.sub_mod(x_mu)  # Q
        
        fft = torch.fft.rfft(x, norm='ortho')
        fft = torch.cat([fft.real, fft.imag], dim=1)
        fft = F.gelu(self.norm2(self.fft_conv(fft)))
        fft_real, fft_imag = torch.chunk(fft, 2, dim=1)        
        fft = torch.complex(fft_real, fft_imag)
        fft = torch.fft.irfft(fft, norm='ortho')

        ts = F.gelu(self.norm1(self.conv(x)))
        
        if self.scm_type == 1:
            y = (x + add_feature - sub_feature) * mul_feature
        elif self.scm_type == 2:
            y = (x + ts + fft)
        elif self.scm_type == 3:
            y = (x + add_feature - sub_feature + ts + fft)
        elif self.scm_type == 4:
            y = (x + add_feature - sub_feature) * mul_feature + ts + fft
        elif self.scm_type == 5:
            y = (x + F.gelu(add_feature - sub_feature)) * mul_feature + ts + fft
        elif self.scm_type == 6:
            y = (x + add_feature - sub_feature + ts + fft) * mul_feature
        elif self.scm_type == 7:
            y = (x + F.gelu(add_feature - sub_feature) + ts + fft) * mul_feature
        elif self.scm_type == 8:
            y = (self.norm1(x) + F.gelu(add_feature - sub_feature)) * mul_feature + ts + fft
            y = F.gelu(y)
            
        elif self.scm_type == 9:
            y = (x + add_feature - sub_feature) * mul_feature
            x_mu_fft = fft - fft.mean([2], keepdim=True)
            
            mul_feature = self.mul_mod(x_mu_fft)  # P
            add_feature = self.add_mod(x_mu_fft)  # K
            sub_feature = self.sub_mod(x_mu_fft)  # Q
            y = y + (x + F.gelu(add_feature - sub_feature)) * mul_feature
            y = y + ts + fft

        elif self.scm_type == 10:
            y = (x + add_feature - sub_feature) * mul_feature
            
            mul_feature = self.mul_mod(fft)  # P
            add_feature = self.add_mod(fft)  # K
            sub_feature = self.sub_mod(fft)  # Q
            y = y + (fft + F.gelu(add_feature - sub_feature)) * mul_feature
            
        elif self.scm_type == 11:
            y = (x + add_feature - sub_feature) * mul_feature
            
            mul_feature = self.mul_mod(fft)  # P
            add_feature = self.add_mod(fft)  # K
            sub_feature = self.sub_mod(fft)  # Q
            y = y + (fft + F.gelu(add_feature - sub_feature)) * mul_feature + ts + fft
            
        elif self.scm_type == 12:
            # y = (x + add_feature - sub_feature) * mul_feature
            mul_feature = self.mul_mod(fft)  # P
            add_feature = self.add_mod(fft)  # K
            sub_feature = self.sub_mod(fft)  # Q
            y = (fft + F.gelu(add_feature - sub_feature)) * mul_feature
            
        if self.se!= False:
            b, c = y.shape[:2]
            z: torch.Tensor = self.avg_pool(y).view(b, c)
            z = self.se(z).view([b, c] + [1] * (y.ndim - 2))
            result = y * z
            result += x
            y = result
            
        return y 

class AttendModule(nn.Module):

    def __init__(self, num_features, num_heads=4):
        super(AttendModule, self).__init__()

        self.num_heads = int(num_heads)
        self.num_features = num_features
        self.num_c_per_head = self.num_features // self.num_heads
        assert self.num_features % self.num_heads == 0

        self.map_gen = nn.Sequential(
            nn.Conv1d(num_features, num_heads, kernel_size=1, stride=1, padding=0, bias=True, groups=num_heads)
        )

        self.normalize = nn.Softmax(dim=2)
        self.return_weight = False

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def batch_weighted_avg(self, xhats, weights):

        b, c, h = xhats.shape
        # xhat reshape
        xhats_reshape = xhats.view(b * self.num_heads, self.num_c_per_head, h)
        # print('xhats_reshape',xhats_reshape.shape)

        # weight reshape
        weights_reshape = weights.view(b * self.num_heads, 1, h)
        # print('weights_reshape',weights_reshape.shape)
        # weights_reshape = weights_reshape.view(b * self.num_heads, 1, h)
        # print('weights_reshape',weights_reshape.shape)

        weights_normalized = self.normalize(weights_reshape)
        # print('weights_normalized',weights_normalized.shape)
        weights_normalized = weights_normalized.transpose(1, 2)
        # print('weights_normalized',weights_normalized.shape)

        mus = torch.bmm(xhats_reshape, weights_normalized)
        # mus = mus.view(b, self.num_heads * self.num_c_per_head, 1, 1)
        # print('mus',mus.shape)
        mus = mus.view(b, self.num_heads * self.num_c_per_head, 1)
        # print('mus',mus.shape)

        return mus, weights_normalized

    def forward(self, x):

        b, c, h = x.shape

        weights = self.map_gen(x)

        mus, weights_normalized = self.batch_weighted_avg(x, weights)

        if self.return_weight:
            weights_normalized = weights_normalized.view(b, self.num_heads, h)
            # weights_normalized = weights_normalized.squeeze(-1)
            # print(weights_normalized.shape)

            # weights_normalized = weights_normalized.view(b, self.num_heads, h)
            weights_splitted = torch.split(weights_normalized, 1, 1)
            # print(weights_normalized.shape, weights_splitted.shape)
            return mus, weights_splitted

        return mus


class ModulateModule(nn.Module):

    def __init__(self, channel, num_groups=32, compressions=2):
        super(ModulateModule, self).__init__()
        self.feature_gen = nn.Sequential(
            nn.Conv1d(channel, channel // compressions, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.GELU(),
            nn.Conv1d(channel // compressions, channel, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.Sigmoid()
        )

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        y = self.feature_gen(x)
        return y
    
