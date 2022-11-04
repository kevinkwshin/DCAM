import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import monai
import random
import numpy as np

from cbam import *
from ffc import *

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    monai.utils.misc.set_determinism(seed=seed)
# pl.seed_everything(seed,True)

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

# modules
class FFT_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FFT_ConvBlock, self).__init__()
        self.img_conv  = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.fft_conv  = nn.Conv1d(in_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.InstanceNorm1d(out_channels)
        self.norm2 = nn.InstanceNorm1d(out_channels*2)

    def forward(self, x):
        # Fourier domain   
        # _, _, W = x.shape
        fft = torch.fft.rfft(x, norm='ortho')
        fft = torch.cat([fft.real, fft.imag], dim=1)
        fft = F.relu(self.norm2(self.fft_conv(fft)))
        fft_real, fft_imag = torch.chunk(fft, 2, dim=1)        
        fft = torch.complex(fft_real, fft_imag)
        fft = torch.fft.irfft(fft, norm='ortho')
        fft = self.norm1(fft)
        # Image domain  
        img = F.leaky_relu(self.norm1(self.img_conv(x)),0.1)

        # Mixing (residual, image, fourier)
        output = x + img + fft
        return output
        
class DFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DFF, self).__init__()
                
        self.conv_fft = torch.nn.Conv1d(in_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0)
        self.norm_fft   = nn.InstanceNorm1d(out_channels*2)
        self.relu   = nn.LeakyReLU(0.1)
        self.channelpool = ChannelPool()
        
        self.conv7 = nn.Conv1d(2, out_channels,kernel_size=7,stride=1,padding=3)

    def forward(self, x):
        print(x.shape)
        x_f = torch.fft.rfft(x, norm='ortho')
        x_f = torch.cat([x_f.real, x_f.imag], dim=1)
        x_f = self.relu(self.norm_fft(self.conv_fft(x_f)))
        print(x_f.shape)
        
        x_f_pool = self.channelpool(x_f)
        print(x_f_cat.shape)
        
        x_f_output = torch.sigmoid(self.conv7(x_f_pool))
        print(x_f_output.shape)        
        
        x_real, x_imag = torch.chunk(x_f, 2, dim=1)
        x_f = torch.complex(x_real, x_imag)
        x_f = torch.fft.irfft(x_f, norm='ortho')
        return x_f

    
class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', dimension=3, norm_layer='batch'):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            norm_layer: whether to add norm ('batch', 'instance', None)
            
            
        # if __name__ == '__main__':
        #     import torch
        #     x = torch.zeros(2, 16, 16)
        #     net = NLBlockND(in_channels=x.shape[1], mode='embedded', dimension=1, norm_layer='instance')
        #     out = net(x)
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            if norm_layer =='batch':
                bn = nn.BatchNorm3d
            elif norm_layer =='instance':
                bn = nn.InstanceNorm3d            
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            if norm_layer =='batch':
                bn = nn.BatchNorm2d
            elif norm_layer =='instance':
                bn = nn.InstanceNorm2d
        else:
            conv_nd = nn.Conv1d
            # conv_nd = nn.Conv1d if FFT==False else FFC
            # conv_nd = nn.Conv1d if FFT==False else FFC_BN_ACT
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            if norm_layer =='batch':
                bn = nn.BatchNorm1d
            elif norm_layer =='instance':
                bn = nn.InstanceNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if norm_layer is not None:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            # nn.init.constant_(self.W_z[1].weight, 0)
            # nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.LeakyReLU(0.1)
                )
            
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
            
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z

import torch
import torch.nn as nn

class ACM(nn.Module):

    def __init__(self, num_heads, num_features, orthogonal_loss=False):
        super(ACM, self).__init__()

        assert num_features % num_heads == 0

        self.num_features = num_features
        self.num_heads = num_heads

        self.add_mod = AttendModule(self.num_features, num_heads=num_heads)
        self.sub_mod = AttendModule(self.num_features, num_heads=num_heads)
        self.mul_mod = ModulateModule(channel=self.num_features, num_groups=num_heads, compressions=2)

        self.orthogonal_loss = orthogonal_loss

        self.init_parameters()

    def init_parameters(self):
        if self.add_mod is not None:
            self.add_mod.init_parameters()
        if self.sub_mod is not None:
            self.sub_mod.init_parameters()
        if self.mul_mod is not None:
            self.mul_mod.init_parameters()

    def forward(self, x):
        
        mu = x.mean([2], keepdim=True)
        x_mu = x - mu

        # creates multipying feature
        mul_feature = self.mul_mod(mu)  # P

        # creates add or sub feature
        add_feature = self.add_mod(x_mu)  # K
        sub_feature = self.sub_mod(x_mu)  # Q
        # print(x_mu.shape,add_feature.shape,sub_feature.shape,mul_feature.shape)
        y = (x + add_feature - sub_feature) * mul_feature

        if self.orthogonal_loss:
            dp = torch.mean(add_feature * sub_feature, dim=1, keepdim=True)
            return y, dp
        else:
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
        xhats_reshape = xhats_reshape.view(b * self.num_heads, self.num_c_per_head, h )

        # weight reshape
        weights_reshape = weights.view(b * self.num_heads, 1, h)
        weights_reshape = weights_reshape.view(b * self.num_heads, 1, h)

        weights_normalized = self.normalize(weights_reshape)
        # print(weights_normalized)
        weights_normalized = weights_normalized.transpose(1, 2)

        mus = torch.bmm(xhats_reshape, weights_normalized)
        # mus = mus.view(b, self.num_heads * self.num_c_per_head, 1, 1)
        mus = mus.view(b, self.num_heads * self.num_c_per_head, 1)

        return mus, weights_normalized

    def forward(self, x):

        b, c, h = x.shape

        weights = self.map_gen(x)

        mus, weights_normalized = self.batch_weighted_avg(x, weights)

        if self.return_weight:
            weights_normalized = weights_normalized.view(b, self.num_heads, h)
            weights_normalized = weights_normalized.squeeze(-1)
            # print(weights_normalized.shape)

            weights_normalized = weights_normalized.view(b, self.num_heads, h)
            weights_splitted = torch.split(weights_normalized, 1, 1)
            # print(weights_splitted.shape)
            return mus, weights_splitted

        return mus


class ModulateModule(nn.Module):

    def __init__(self, channel, num_groups=32, compressions=2):
        super(ModulateModule, self).__init__()
        self.feature_gen = nn.Sequential(
            nn.Conv1d(channel, channel // compressions, kernel_size=1, stride=1, padding=0, bias=True,
                      groups=num_groups),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // compressions, channel, kernel_size=1, stride=1, padding=0, bias=True,
                      groups=num_groups),
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

# # if __name__ == '__main__':
# #     x1 = torch.randn(256 * 20 * 20 * 5).view(5, 256, 20, 20).float()
# #     x1 = torch.rand(2, 320, 160).float()
# #     acm = ACM(num_heads=32, num_features=320, orthogonal_loss=True)
# #     acm.init_parameters()
# #     y, dp = acm(x1)
# #     print(y.shape)
# #     print(dp.shape)

# #     ACM without orthogonal loss
# #     acm = ACM(num_heads=32, num_features=320, orthogonal_loss=False)
# #     acm.init_parameters()
# #     y = acm(x1)
# #     print(x1.shape,y.shape)

# import torch
# import torch.nn as nn
# from torch.utils.model_zoo import load_url as load_state_dict_from_url
# # import context_module

# num_acm_groups = 16

# model_urls = {
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#     'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#     'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
#     'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
# }


# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)

# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1
#     __constants__ = ['downsample']

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None, module="none"):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm1d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride

#         if module:
#             print("not implemented yet")
#             raise ValueError

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4
#     __constants__ = ['downsample']

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, module="none", orthogonal_loss=False):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm1d
#         width = int(planes * (base_width / 64.)) * groups

#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.module_type = module

#         if module == "none":
#             self.module = None
#         elif module == 'acm':
#             self.module = ACM(num_heads=num_acm_groups, num_features=planes * 4, orthogonal_loss=orthogonal_loss)
#             self.module.init_parameters()
#         else:
#             raise ValueError("undefined module")

#     def forward(self, x):

#         if isinstance(x, tuple):
#             x, prev_dp = x
#         else:
#             prev_dp = None

#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)

#         out = self.bn2(out)

#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         dp = None
#         if self.module is not None:
#             out = self.module(out)
#             if isinstance(out, tuple):
#                 out, dp = out
#                 if prev_dp is not None:
#                     dp = prev_dp + dp

#         out += identity
#         out = self.relu(out)

#         if dp is None:
#             return out
#         else:
#             # diff loss
#             return out, dp

# class ResNet(nn.Module):

#     def __init__(self, block, layers, in_channels=3, num_classes=1000, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None, module=""):
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm1d
#         self._norm_layer = norm_layer

#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv1d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         assert module in ['none', 'acm']
#         self.layer1 = self._make_layer(block, 64,  layers[0], 1, dilate=False,                           module=module)
#         self.layer2 = self._make_layer(block, 128, layers[1], 2, dilate=replace_stride_with_dilation[0], module=module)
#         self.layer3 = self._make_layer(block, 256, layers[2], 2, dilate=replace_stride_with_dilation[1], module=module)
#         self.layer4 = self._make_layer(block, 512, layers[3], 2, dilate=replace_stride_with_dilation[2], module=module)
#         # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         # self.fc = nn.Linear(512 * block.expansion, num_classes)
#         self.classifier_conv =nn.Conv1d(512 * block.expansion, num_classes, kernel_size=1)
#         self.classifier_maxpool= nn.AdaptiveMaxPool1d(1)

#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm)):
#                 try:
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#                 except:
#                     print("Module without affine doesnt have weights")

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)

#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False, module=""):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []

#         # configure module placement
#         module_placement = [module] * blocks
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer, module=module_placement[0]))
#         self.inplanes = planes * block.expansion
#         for block_idx in range(1, blocks):

#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer, module=module_placement[block_idx]))

#         return nn.Sequential(*layers)

#     def _forward_body(self, x):

#         x0 = self.conv1(x)
#         x0 = self.bn1(x0)
#         x0 = self.relu(x0)
#         x0 = self.maxpool(x0)

#         x1 = self.layer1(x0)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)
#         return x0,x1,x2,x3,x4
    
#     def _forward_task(self, x):
    
# #         x = self.avgpool(x)
# #         x = x.squeeze()
# #         x = self.fc(x)
# #         x = nn.Sigmoid()(x)
        
#         x = self.classifier_conv(x)
#         x = self.classifier_maxpool(x)
#         x = x.squeeze()
#         x = nn.Sigmoid()(x)
        
#         return x

#     def forward(self, x):
#         x0,x1,x2,x3,x4 = self._forward_body(x)
#         if isinstance(x4, tuple):
#             x4, dp = x4
#         else:
#             dp = None
#         logit = self._forward_task(x4)
#         if dp is not None:
#             return logit, dp
#         else:
#             return logit        
        
# class ResNetFeature(nn.Module):
#     def __init__(self, block, layers, in_channels=3, num_classes=1000, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None, module=""):
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm1d
#         self._norm_layer = norm_layer

#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv1d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         assert module in ['none', 'acm']
#         self.layer1 = self._make_layer(block, 64,  layers[0], 1, dilate=False,                           module=module)
#         self.layer2 = self._make_layer(block, 128, layers[1], 2, dilate=replace_stride_with_dilation[0], module=module)
#         self.layer3 = self._make_layer(block, 256, layers[2], 2, dilate=replace_stride_with_dilation[1], module=module)
#         self.layer4 = self._make_layer(block, 512, layers[3], 2, dilate=replace_stride_with_dilation[2], module=module)

#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm)):
#                 try:
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#                 except:
#                     print("Module without affine doesnt have weights")

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)

#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False, module=""):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []

#         # configure module placement
#         module_placement = [module] * blocks
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer, module=module_placement[0]))
#         self.inplanes = planes * block.expansion
#         for block_idx in range(1, blocks):

#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer, module=module_placement[block_idx]))

#         return nn.Sequential(*layers)

#     def _forward_body(self, x):

#         x0 = self.conv1(x)
#         x0 = self.bn1(x0)
#         x0 = self.relu(x0)
#         x0 = self.maxpool(x0)

#         x1 = self.layer1(x0)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)
#         return x0,x1,x2,x3,x4
    
#     def _forward_task(self, x):
    
# #         x = self.avgpool(x)
# #         x = x.squeeze()
# #         x = self.fc(x)
# #         x = nn.Sigmoid()(x)
        
#         x = self.classifier_conv(x)
#         x = self.classifier_maxpool(x)
#         x = x.squeeze()
#         x = nn.Sigmoid()(x)
        
#         return x

#     def forward(self, x):
#         x0,x1,x2,x3,x4 = self._forward_body(x)
#         if isinstance(x4, tuple):
#             x4, dp = x4
#         else:
#             dp = None
#         if dp is not None:
#             return x0,x1,x2,x3,x4, dp
#         else:
#             return x0,x1,x2,x3,x4       

# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
#         try:
#             model.load_state_dict(state_dict)
#         except:
#             print("keys did not mathc")
#             print("entering custom loading statedict")
#             existing_statedict = model.state_dict()
#             cooccuring = {}
#             for key, val in state_dict.items():
#                 if key in existing_statedict:
#                     cooccuring[key] = state_dict[key]
#                 else:
#                     print(key, "does not exists in the new model")
#             print("keys adding ", len(cooccuring.keys()))
#             print("whole keys", len(existing_statedict.keys()))
#             existing_statedict.update(cooccuring)
#             model.load_state_dict(existing_statedict)

#     return model


# def resnet34(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-34 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)


# def resnet50(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-50 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)


# def resnet101(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-101 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
#                    **kwargs)


# def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-50 32x4d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)

# def resnet(arch, pretrained, in_channels, num_classes, zero_init_residual, module):
#     """
#     # # 1D model
#     # # net = resnet('resnet101',False,1,2,False,'acm',32)
#     # net = resnet('resnet101',False,1,2, False,'none',32)

#     # x = torch.rand(2,1,64)
#     # yhat = net(x)

#     # if isinstance(yhat,tuple):
#     #     for yh in yhat:
#     #         print(yh.shape)
#     # else:
#     #     print(yhat.shape)
#     """
#     if arch == "resnet50":
#         model = resnet50(pretrained=pretrained, progress=True, in_channels=in_channels, num_classes=num_classes,
#                          zero_init_residual=zero_init_residual, module=module)
#     elif arch == "resnet101":
#         model = resnet101(pretrained=pretrained, progress=True, in_channels=in_channels, num_classes=num_classes,
#                          zero_init_residual=zero_init_residual, module=module)
#     elif arch == "resnext50_32x4d":
#         model = resnet101(pretrained=pretrained, progress=True, in_channels=in_channels, num_classes=num_classes,
#                          zero_init_residual=zero_init_residual, module=module)
#     else:
#         raise ValueError
#     return model


# class REBNCONV(nn.Module):
#     def __init__(self,in_ch=1,out_ch=1,dirate=1,dropout=0.1,norm='instance'):
#         super(REBNCONV,self).__init__()

#         self.conv_s1 = nn.Conv1d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
#         if norm=='instance':
#             self.bn_s1 = nn.InstanceNorm1d(out_ch)
#         elif norm=='batch':
#             self.bn_s1 = nn.BatchNorm1d(out_ch)    
#         self.relu_s1 = nn.LeakyReLU(0.1)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self,x):

#         hx = x
#         xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
#         xout = self.dropout(xout)
        
#         return xout

# ## upsample tensor 'src' to have the same spatial size with tensor 'tar'
# def _upsample_like(src,tar):
#     src = F.upsample(src,size=tar.shape[2:],mode='linear')

#     return src


# ### RSU-7 ###
# class RSU7(nn.Module):#UNet07DRES(nn.Module):

#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, dropout=0.1, norm='instance'):
#         super(RSU7,self).__init__()

#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1,dropout=dropout,norm= norm)

#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1,dropout=dropout,norm= norm)
#         self.pool1 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout,norm= norm)
#         self.pool2 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout,norm= norm)
#         self.pool3 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout,norm= norm)
#         self.pool4 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout,norm= norm)
#         self.pool5 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout,norm= norm)

#         self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2,dropout=dropout,norm= norm)

#         self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout,norm= norm)
#         self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout,norm= norm)
#         self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout,norm= norm)
#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout,norm= norm)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout,norm= norm)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1,dropout=dropout,norm= norm)

#     def forward(self,x):

#         hx = x
#         hxin = self.rebnconvin(hx)

#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)

#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)

#         hx3 = self.rebnconv3(hx)
#         hx = self.pool3(hx3)

#         hx4 = self.rebnconv4(hx)
#         hx = self.pool4(hx4)

#         hx5 = self.rebnconv5(hx)
#         hx = self.pool5(hx5)

#         hx6 = self.rebnconv6(hx)

#         hx7 = self.rebnconv7(hx6)

#         hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
#         hx6dup = _upsample_like(hx6d,hx5)

#         hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
#         hx5dup = _upsample_like(hx5d,hx4)

#         hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
#         hx4dup = _upsample_like(hx4d,hx3)

#         hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)

#         hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)

#         hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

#         return hx1d + hxin

# ### RSU-6 ###
# class RSU6(nn.Module):#UNet06DRES(nn.Module):

#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, dropout=0.1, norm='instance'):
#         super(RSU6,self).__init__()

#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1,dropout=dropout, norm=norm)

#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)
#         self.pool1 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)
#         self.pool2 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)
#         self.pool3 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)
#         self.pool4 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)

#         self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2,dropout=dropout, norm=norm)

#         self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm=norm)
#         self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm=norm)
#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm=norm)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm=norm)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1,dropout=dropout, norm=norm)

#     def forward(self,x):

#         hx = x

#         hxin = self.rebnconvin(hx)

#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)

#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)

#         hx3 = self.rebnconv3(hx)
#         hx = self.pool3(hx3)

#         hx4 = self.rebnconv4(hx)
#         hx = self.pool4(hx4)

#         hx5 = self.rebnconv5(hx)

#         hx6 = self.rebnconv6(hx5)


#         hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
#         hx5dup = _upsample_like(hx5d,hx4)

#         hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
#         hx4dup = _upsample_like(hx4d,hx3)

#         hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)

#         hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)

#         hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

#         return hx1d + hxin

# ### RSU-5 ###
# class RSU5(nn.Module):#UNet05DRES(nn.Module):

#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3,dropout=0.1,norm='instance'):
#         super(RSU5,self).__init__()

#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1,dropout=dropout, norm= norm)

#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1,dropout=dropout, norm= norm)
#         self.pool1 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm= norm)
#         self.pool2 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm= norm)
#         self.pool3 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm= norm)

#         self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2,dropout=dropout, norm= norm)

#         self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm= norm)
#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm= norm)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm= norm)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1,dropout=dropout, norm= norm)

#     def forward(self,x):

#         hx = x

#         hxin = self.rebnconvin(hx)

#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)

#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)

#         hx3 = self.rebnconv3(hx)
#         hx = self.pool3(hx3)

#         hx4 = self.rebnconv4(hx)

#         hx5 = self.rebnconv5(hx4)

#         hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
#         hx4dup = _upsample_like(hx4d,hx3)

#         hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)

#         hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)

#         hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

#         return hx1d + hxin

# ### RSU-4 ###
# class RSU4(nn.Module):#UNet04DRES(nn.Module):

#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3,dropout=0.1,norm='instance'):
#         super(RSU4,self).__init__()

#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1,dropout=dropout, norm=norm)

#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)
#         self.pool1 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)
#         self.pool2 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)

#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2,dropout=dropout, norm=norm)

#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm=norm)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm=norm)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1,dropout=dropout, norm=norm)

#     def forward(self,x):

#         hx = x

#         hxin = self.rebnconvin(hx)

#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)

#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)

#         hx3 = self.rebnconv3(hx)

#         hx4 = self.rebnconv4(hx3)

#         hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)

#         hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)

#         hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

#         return hx1d + hxin

# ### RSU-4F ###
# class RSU4F(nn.Module):#UNet04FRES(nn.Module):

#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3,dropout=0.1, norm='instance'):
#         super(RSU4F,self).__init__()

#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1,dropout=dropout, norm= norm)

#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1,dropout=dropout, norm= norm)
#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2,dropout=dropout, norm= norm)
#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4,dropout=dropout, norm= norm)

#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8,dropout=dropout, norm= norm)

#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4,dropout=dropout, norm= norm)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2,dropout=dropout, norm= norm)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1,dropout=dropout, norm= norm)

#     def forward(self,x):

#         hx = x

#         hxin = self.rebnconvin(hx)

#         hx1 = self.rebnconv1(hxin)
#         hx2 = self.rebnconv2(hx1)
#         hx3 = self.rebnconv3(hx2)

#         hx4 = self.rebnconv4(hx3)

#         hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
#         hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
#         hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

#         return hx1d + hxin


# ##### U^2-Net ####
# class U2NET(nn.Module):

#     def __init__(self,in_ch=1,out_ch=1,nnblock=False, FFC=False, acm=False, ASPP=False, temperature=1, dropout=0.1, norm='instance'):
#         super(U2NET,self).__init__()

#         self.stage1 = RSU7(in_ch,32,64,dropout=dropout, norm= norm)
#         self.pool12 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.stage2 = RSU6(64,32,128,dropout=dropout, norm= norm)
#         self.pool23 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.stage3 = RSU5(128,64,256,dropout=dropout, norm= norm)
#         self.pool34 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.stage4 = RSU4(256,128,512,dropout=dropout, norm= norm)
#         self.pool45 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.stage5 = RSU4F(512,256,512,dropout=dropout, norm= norm)
#         self.pool56 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.stage6 = RSU4F(512,256,512,dropout=dropout, norm= norm)

#         # decoder
#         self.stage5d = RSU4F(1024,256,512,dropout=dropout, norm= norm)
#         self.stage4d = RSU4(1024,128,256,dropout=dropout, norm= norm)
#         self.stage3d = RSU5(512,64,128,dropout=dropout, norm= norm)
#         self.stage2d = RSU6(256,32,64,dropout=dropout, norm= norm)
#         self.stage1d = RSU7(128,16,64,dropout=dropout, norm= norm)

#         self.side1 = nn.Conv1d(64,out_ch,3,padding=1)
#         self.side2 = nn.Conv1d(64,out_ch,3,padding=1)
#         self.side3 = nn.Conv1d(128,out_ch,3,padding=1)
#         self.side4 = nn.Conv1d(256,out_ch,3,padding=1)
#         self.side5 = nn.Conv1d(512,out_ch,3,padding=1)
#         self.side6 = nn.Conv1d(512,out_ch,3,padding=1)

#         self.outconv = nn.Conv1d(6*out_ch,out_ch,1)
        
#         fea = [64, 128, 256, 512, 512, 512]
#         self.nnblock = nnblock
#         if nnblock:
#             spatial_dims = 1
#             self.nnblock1 = NLBlockND(in_channels=fea[0], mode='embedded', dimension=spatial_dims, norm_layer=norm)
#             self.nnblock2 = NLBlockND(in_channels=fea[1], mode='embedded', dimension=spatial_dims, norm_layer=norm)
#             self.nnblock3 = NLBlockND(in_channels=fea[2], mode='embedded', dimension=spatial_dims, norm_layer=norm)
#             self.nnblock4 = NLBlockND(in_channels=fea[3], mode='embedded', dimension=spatial_dims, norm_layer=norm)
#             self.nnblock5 = NLBlockND(in_channels=fea[4], mode='embedded', dimension=spatial_dims, norm_layer=norm)     
#             self.nnblock6 = NLBlockND(in_channels=fea[5], mode='embedded', dimension=spatial_dims, norm_layer=norm)                      

#         self.FFC = FFC
#         if FFC=='FFC':
#             self.FFCblock1 = FFC_BN_ACT(fea[0],fea[0])
#             self.FFCblock2 = FFC_BN_ACT(fea[1],fea[1])
#             self.FFCblock3 = FFC_BN_ACT(fea[2],fea[2])
#             self.FFCblock4 = FFC_BN_ACT(fea[3],fea[3])
#             self.FFCblock5 = FFC_BN_ACT(fea[4],fea[4])
#             self.FFCblock6 = FFC_BN_ACT(fea[5],fea[5])            
#         elif FFC=='DeepRFT':
#             self.FFCblock1 = FFT_ConvBlock(fea[0],fea[0])
#             self.FFCblock2 = FFT_ConvBlock(fea[1],fea[1])
#             self.FFCblock3 = FFT_ConvBlock(fea[2],fea[2])
#             self.FFCblock4 = FFT_ConvBlock(fea[3],fea[3])
#             self.FFCblock5 = FFT_ConvBlock(fea[4],fea[4])
#             self.FFCblock6 = FFT_ConvBlock(fea[5],fea[5])
            
#         self.acm = acm
#         if acm:
#             self.acm1 = ACM(num_heads=fea[0]//2, num_features=fea[0], orthogonal_loss=False)
#             self.acm2 = ACM(num_heads=fea[1]//2, num_features=fea[1], orthogonal_loss=False)
#             self.acm3 = ACM(num_heads=fea[2]//2, num_features=fea[2], orthogonal_loss=False)
#             self.acm4 = ACM(num_heads=fea[3]//2, num_features=fea[3], orthogonal_loss=False)
#             self.acm5 = ACM(num_heads=fea[4]//2, num_features=fea[4], orthogonal_loss=False)
#             self.acm6 = ACM(num_heads=fea[5]//2, num_features=fea[5], orthogonal_loss=False)
            
#         self.ASPP = ASPP
#         spatial_dims = 1
#         if ASPP=='last':
#             self.ASPPblock6 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[5], conv_out_channels=fea[5]//4,
#                                                                norm_type=norm, acti_type='LEAKYRELU', bias=False)  
#         elif ASPP=='all':
#             self.ASPPblock1 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[0], conv_out_channels=fea[0]//4,
#                                                                norm_type=norm, acti_type='LEAKYRELU', bias=False)  
#             self.ASPPblock2 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[1], conv_out_channels=fea[1]//4,
#                                                                norm_type=norm, acti_type='LEAKYRELU', bias=False)  
#             self.ASPPblock3 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[2], conv_out_channels=fea[2]//4,
#                                                                norm_type=norm, acti_type='LEAKYRELU', bias=False)  
#             self.ASPPblock4 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[3], conv_out_channels=fea[3]//4,
#                                                                norm_type=norm, acti_type='LEAKYRELU', bias=False) 
#             self.ASPPblock5 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[4], conv_out_channels=fea[4]//4,
#                                                                norm_type=norm, acti_type='LEAKYRELU', bias=False)  
#             self.ASPPblock6 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[5], conv_out_channels=fea[5]//4,
#                                                                norm_type=norm, acti_type='LEAKYRELU', bias=False)  
#         self.temperature = temperature
            
#     def forward(self,x):

#         hx = x

#         #stage 1
#         hx1 = self.stage1(hx)
#         hx1 = hx1 + self.ASPPblock1(hx1) if self.ASPP=='all' else hx1
#         hx1 = hx1 + self.nnblock1(hx1) if self.nnblock else hx1
#         hx1 = hx1 + self.FFCblock1(hx1) if self.FFC else hx1
#         hx1 = hx1 + self.acm1(hx1) if self.acm else hx1
#         hx = self.pool12(hx1)

#         #stage 2
#         hx2 = self.stage2(hx)
#         hx2 = hx2 + self.ASPPblock2(hx2) if self.ASPP=='all' else hx2
#         hx2 = hx2 + self.nnblock2(hx2) if self.nnblock else hx2
#         hx2 = hx2 + self.FFCblock2(hx2) if self.FFC else hx2
#         hx2 = hx2 + self.acm2(hx2) if self.acm else hx2
#         hx = self.pool23(hx2)

#         #stage 3
#         hx3 = self.stage3(hx)
#         hx3 = hx3 + self.ASPPblock3(hx3) if self.ASPP=='all' else hx3
#         hx3 = hx3 + self.nnblock3(hx3) if self.nnblock else hx3
#         hx3 = hx3 + self.FFCblock3(hx3) if self.FFC else hx3
#         hx3 = hx3 + self.acm3(hx3) if self.acm else hx3
#         hx = self.pool34(hx3)

#         #stage 4
#         hx4 = self.stage4(hx)
#         hx4 = hx4 + self.ASPPblock4(hx4) if self.ASPP=='all' else hx4
#         hx4 = hx4 + self.nnblock4(hx4) if self.nnblock else hx4
#         hx4 = hx4 + self.FFCblock4(hx4) if self.FFC else hx4
#         hx4 = hx4 + self.acm4(hx4) if self.acm else hx4
#         hx = self.pool45(hx4)

#         #stage 5
#         hx5 = self.stage5(hx)
#         hx5 = hx5 + self.ASPPblock5(hx5) if self.ASPP=='all' else hx5
#         hx5 = hx5 +self.nnblock5(hx5) if self.nnblock else hx5
#         hx5 = hx5 +self.FFCblock5(hx5) if self.FFC else hx5
#         hx5 = hx5 +self.acm5(hx5) if self.acm else hx5
#         hx = self.pool56(hx5)

#         #stage 6
#         hx6 = self.stage6(hx)
#         hx6 = hx6 + self.ASPPblock6(hx6) if self.ASPP else hx6
#         hx6 = hx6 + self.nnblock6(hx6) if self.nnblock else hx6
#         hx6 = hx6 + self.FFCblock6(hx6) if self.FFC else hx6
#         hx6 = hx6 + self.acm6(hx6) if self.acm else hx6
#         hx6up = _upsample_like(hx6,hx5)
        
#         #-------------------- decoder --------------------
#         hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
#         hx5dup = _upsample_like(hx5d,hx4)

#         hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
#         hx4dup = _upsample_like(hx4d,hx3)

#         hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)

#         hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)

#         hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


#         #side output
#         d1 = self.side1(hx1d)

#         d2 = self.side2(hx2d)
#         d2 = _upsample_like(d2,d1)

#         d3 = self.side3(hx3d)
#         d3 = _upsample_like(d3,d1)

#         d4 = self.side4(hx4d)
#         d4 = _upsample_like(d4,d1)

#         d5 = self.side5(hx5d)
#         d5 = _upsample_like(d5,d1)

#         d6 = self.side6(hx6)
#         d6 = _upsample_like(d6,d1)

#         d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

#         return torch.sigmoid(d0/self.temperature), torch.sigmoid(d1/self.temperature), torch.sigmoid(d2/self.temperature), torch.sigmoid(d3/self.temperature), torch.sigmoid(d4/self.temperature), torch.sigmoid(d5/self.temperature), torch.sigmoid(d6/self.temperature)

# ### U^2-Net small ###
# class U2NETP(nn.Module):

#     def __init__(self,in_ch=3,out_ch=1):
#         super(U2NETP,self).__init__()

#         self.stage1 = RSU7(in_ch,16,64)
#         self.pool12 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.stage2 = RSU6(64,16,64)
#         self.pool23 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.stage3 = RSU5(64,16,64)
#         self.pool34 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.stage4 = RSU4(64,16,64)
#         self.pool45 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.stage5 = RSU4F(64,16,64)
#         self.pool56 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

#         self.stage6 = RSU4F(64,16,64)

#         # decoder
#         self.stage5d = RSU4F(128,16,64)
#         self.stage4d = RSU4(128,16,64)
#         self.stage3d = RSU5(128,16,64)
#         self.stage2d = RSU6(128,16,64)
#         self.stage1d = RSU7(128,16,64)

#         self.side1 = nn.Conv1d(64,out_ch,3,padding=1)
#         self.side2 = nn.Conv1d(64,out_ch,3,padding=1)
#         self.side3 = nn.Conv1d(64,out_ch,3,padding=1)
#         self.side4 = nn.Conv1d(64,out_ch,3,padding=1)
#         self.side5 = nn.Conv1d(64,out_ch,3,padding=1)
#         self.side6 = nn.Conv1d(64,out_ch,3,padding=1)

#         self.outconv = nn.Conv1d(6*out_ch,out_ch,1)

#     def forward(self,x):

#         hx = x

#         #stage 1
#         hx1 = self.stage1(hx)
#         hx = self.pool12(hx1)

#         #stage 2
#         hx2 = self.stage2(hx)
#         hx = self.pool23(hx2)

#         #stage 3
#         hx3 = self.stage3(hx)
#         hx = self.pool34(hx3)

#         #stage 4
#         hx4 = self.stage4(hx)
#         hx = self.pool45(hx4)

#         #stage 5
#         hx5 = self.stage5(hx)
#         hx = self.pool56(hx5)

#         #stage 6
#         hx6 = self.stage6(hx)
#         hx6up = _upsample_like(hx6,hx5)

#         #decoder
#         hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
#         hx5dup = _upsample_like(hx5d,hx4)

#         hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
#         hx4dup = _upsample_like(hx4d,hx3)

#         hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)

#         hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)

#         hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


#         #side output
#         d1 = self.side1(hx1d)

#         d2 = self.side2(hx2d)
#         d2 = _upsample_like(d2,d1)

#         d3 = self.side3(hx3d)
#         d3 = _upsample_like(d3,d1)

#         d4 = self.side4(hx4d)
#         d4 = _upsample_like(d4,d1)

#         d5 = self.side5(hx5d)
#         d5 = _upsample_like(d5,d1)

#         d6 = self.side6(hx6)
#         d6 = _upsample_like(d6,d1)

#         d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

#         return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

# bce_loss = nn.BCELoss()

# def muti_bce_loss_fusion(yhat, y):
#     d0, d1, d2, d3, d4, d5, d6 = yhat
#     labels_v = y
#     loss0 = bce_loss(d0,labels_v)
#     loss1 = bce_loss(d1,labels_v)
#     loss2 = bce_loss(d2,labels_v)
#     loss3 = bce_loss(d3,labels_v)
#     loss4 = bce_loss(d4,labels_v)
#     loss5 = bce_loss(d5,labels_v)
#     loss6 = bce_loss(d6,labels_v)

#     loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
#     return loss

# lossfn = muti_bce_loss_fusion

from typing import Optional, Sequence, Union

from monai.networks.blocks import Convolution, UpSample
from monai.networks.nets.basic_unet import TwoConv, Down, UpCat, UpSample, Union
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

import nets
from nets import *

from monai.networks.nets import ResNet, DenseNet, SENet
from monai.networks.nets.resnet import ResNetBlock#, ResNetBottleneck
from monai.networks.nets.senet import SEBottleneck, SEResNetBottleneck
import torch.nn.functional as F 
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Type, Union
import torch
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, Pool

class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: Union[nn.Module, partial, None] = None,
        # module="acm"
        module="none"
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels (taking expansion into account).
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for second conv layer.
            downsample: which downsample layer to use.
        """

        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        # norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        norm_type: Callable = Norm[Norm.INSTANCE, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_type(planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_type(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        num_acm_groups = 32
        orthogonal_loss= False
        
        if module == "none":
            self.module = None
        elif module == 'acm':
            self.module = nets.ACM(num_heads=num_acm_groups, num_features=planes * 4, orthogonal_loss=orthogonal_loss)
            self.module.init_parameters()
        else:
            raise ValueError("undefined module")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if isinstance(x, tuple):
            x, prev_dp = x
        else:
            prev_dp = None

        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            
        dp = None
        if self.module is not None:
            out = self.module(out)
            if isinstance(out, tuple):
                out, dp = out
                if prev_dp is not None:
                    dp = prev_dp + dp

        out += residual
        out = self.relu(out)

        if dp is None:
            return out
        else:
            # diff loss
            return out, dp
        
class ResNet(nn.Module):
    """
    ResNet based on: `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? <https://arxiv.org/pdf/1711.09577.pdf>`_.
    Adapted from `<https://github.com/kenshohara/3D-ResNets-PyTorch/tree/master/models>`_.

    Args:
        block: which ResNet block to use, either Basic or Bottleneck.
            ResNet block class or str.
            for Basic: ResNetBlock or 'basic'
            for Bottleneck: ResNetBottleneck or 'bottleneck'
        layers: how many layers to use.
        block_inplanes: determine the size of planes at each step. Also tunable with widen_factor.
        spatial_dims: number of spatial dimensions of the input image.
        n_input_channels: number of input channels for first convolutional layer.
        conv1_t_size: size of first convolution layer, determines kernel and padding.
        conv1_t_stride: stride of first convolution layer.
        no_max_pool: bool argument to determine if to use maxpool layer.
        shortcut_type: which downsample block to use. Options are 'A', 'B', default to 'B'.
            - 'A': using `self._downsample_basic_block`.
            - 'B': kernel_size 1 conv + norm.
        widen_factor: widen output for each layer.
        num_classes: number of output (classifications).
        feed_forward: whether to add the FC layer for the output, default to `True`.

    """

    def __init__(
        self,
        block: Union[Type[Union[ResNetBlock, ResNetBottleneck]], str],
        layers: List[int],
        block_inplanes: List[int],
        spatial_dims: int = 3,
        n_input_channels: int = 3,
        conv1_t_size: Union[Tuple[int], int] = 7,
        conv1_t_stride: Union[Tuple[int], int] = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        num_classes: int = 400,
        feed_forward: bool = True,
        norm = 'instance'
    ) -> None:

        super().__init__()

        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            elif block == "bottleneck":
                block = ResNetBottleneck
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)

        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        if norm == 'batch':
            norm_type: Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        else:
            norm_type: Type[Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]] = Norm[Norm.INSTANCE, spatial_dims]
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        avgp_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        block_avgpool = get_avgpool()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        conv1_kernel_size = ensure_tuple_rep(conv1_t_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(conv1_t_stride, spatial_dims)

        self.conv1 = conv_type(
            n_input_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,  # type: ignore
            stride=conv1_stride,  # type: ignore
            padding=tuple(k // 2 for k in conv1_kernel_size),  # type: ignore
            bias=False,
        )
        self.bn1 = norm_type(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = pool_type(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], spatial_dims, shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], spatial_dims, shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=2)
        self.avgpool = avgp_type(block_avgpool[spatial_dims])
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_classes) if feed_forward else None

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int, spatial_dims: int = 3) -> torch.Tensor:
        out: torch.Tensor = get_pool_layer(("avg", {"kernel_size": 1, "stride": stride}), spatial_dims=spatial_dims)(x)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(
        self,
        block: Type[Union[ResNetBlock, ResNetBottleneck]],
        planes: int,
        blocks: int,
        spatial_dims: int,
        shortcut_type: str,
        stride: int = 1,
    ) -> nn.Sequential:

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        downsample: Union[nn.Module, partial, None] = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if look_up_option(shortcut_type, {"A", "B"}) == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    spatial_dims=spatial_dims,
                )
            else:
                downsample = nn.Sequential(
                    conv_type(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride),
                    norm_type(planes * block.expansion),
                )

        layers = [
            block(
                in_planes=self.in_planes, planes=planes, spatial_dims=spatial_dims, stride=stride, downsample=downsample
            )
        ]

        self.in_planes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.fc is not None:
            x = self.fc(x)

        return x
    
class ResNetFeature(ResNet):
    def __init__(
            self,
            block: Union[Type[Union[ResNetBlock, ResNetBottleneck]], str],
            layers: List[int],
            block_inplanes: List[int],
            spatial_dims: int = 3,
            n_input_channels: int = 3,
            conv1_t_size: Union[Tuple[int], int] = 7,
            conv1_t_stride: Union[Tuple[int], int] = 1,
            no_max_pool: bool = False,
            shortcut_type: str = "B",
            widen_factor: float = 1.0,
            num_classes: int = 400,
            feed_forward: bool = True,
            n_classes: Optional[int] = None,
            norm='instance'
        ) -> None:
        super().__init__(block,layers,block_inplanes,spatial_dims,n_input_channels,conv1_t_size,conv1_t_stride,no_max_pool)
        self.spatial_dims= spatial_dims
    
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        self.mpool1 = pool_type(kernel_size=3, stride=2, ceil_mode=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        if not self.no_max_pool:
            x1 = self.maxpool(x1)
        x2 = self.layer1(x1)        
        x2 = self.mpool1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5

# def bn2instance(module):
#     module_output = module
#     if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
#         module_output = torch.nn.InstanceNorm1d(module.num_features,
#                                                 module.eps, module.momentum,
#                                                 module.affine,
#                                                 module.track_running_stats)
#         if module.affine:
#             with torch.no_grad():
#                 module_output.weight = module.weight
#                 module_output.bias = module.bias
#         module_output.running_mean = module.running_mean
#         module_output.running_var = module.running_var
#         module_output.num_batches_tracked = module.num_batches_tracked
#         if hasattr(module, "qconfig"):
#             module_output.qconfig = module.qconfig

#     for name, child in module.named_children():
#         module_output.add_module(name, bn2instance(child))

#     del module
#     return module_output

def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:], mode='linear')
    return src

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import operator
import re
from functools import reduce
from typing import List, NamedTuple, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.utils import model_zoo

from monai.networks.layers.factories import Act, Conv, Pad, Pool
from monai.networks.layers.utils import get_norm_layer
from monai.utils.module import look_up_option
from typing import Optional, Sequence, Union

from monai.networks.blocks import Convolution, UpSample
from monai.networks.nets.basic_unet import TwoConv, Down, UpCat, UpSample, Union
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

import nets
from nets import *

from monai.networks.nets import ResNet, DenseNet, SENet
from monai.networks.nets.resnet import ResNetBlock#, ResNetBottleneck
from monai.networks.nets.senet import SEBottleneck, SEResNetBottleneck
import torch.nn.functional as F 
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Type, Union
import torch
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, Pool

def get_inplanes():
    return [64, 128, 256, 512]


def get_avgpool():
    return [0, 1, (1, 1), (1, 1, 1)]


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: Union[nn.Module, partial, None] = None,
        # module="acm"
        module="none"
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels (taking expansion into account).
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for second conv layer.
            downsample: which downsample layer to use.
        """

        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        # norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        norm_type: Callable = Norm[Norm.INSTANCE, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_type(planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_type(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        num_acm_groups = 32
        orthogonal_loss= False
        
        if module == "none":
            self.module = None
        elif module == 'acm':
            self.module = nets.ACM(num_heads=num_acm_groups, num_features=planes * 4, orthogonal_loss=orthogonal_loss)
            self.module.init_parameters()
        else:
            raise ValueError("undefined module")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if isinstance(x, tuple):
            x, prev_dp = x
        else:
            prev_dp = None

        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            
        dp = None
        if self.module is not None:
            out = self.module(out)
            if isinstance(out, tuple):
                out, dp = out
                if prev_dp is not None:
                    dp = prev_dp + dp

        out += residual
        out = self.relu(out)

        if dp is None:
            return out
        else:
            # diff loss
            return out, dp

class ResNetFeature(ResNet):
    def __init__(
            self,
            block: Union[Type[Union[ResNetBlock, ResNetBottleneck]], str],
            layers: List[int],
            block_inplanes: List[int],
            spatial_dims: int = 3,
            n_input_channels: int = 3,
            conv1_t_size: Union[Tuple[int], int] = 7,
            conv1_t_stride: Union[Tuple[int], int] = 1,
            no_max_pool: bool = False,
            shortcut_type: str = "B",
            widen_factor: float = 1.0,
            num_classes: int = 400,
            feed_forward: bool = True,
            n_classes: Optional[int] = None,
        ) -> None:
        super().__init__(block,layers,block_inplanes,spatial_dims,n_input_channels,conv1_t_size,conv1_t_stride,no_max_pool)
        self.spatial_dims= spatial_dims
    
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        self.mpool1 = pool_type(kernel_size=3, stride=2, ceil_mode=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        if not self.no_max_pool:
            x1 = self.maxpool(x1)
        x2 = self.layer1(x1)        
        x2 = self.mpool1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5

def _resnet(
    arch: str,
    block: Type[Union[ResNetBlock, ResNetBottleneck]],
    layers: List[int],
    block_inplanes: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model: ResNetFeature = ResNetFeature(block, layers, block_inplanes, **kwargs)
    if pretrained:
        # Author of paper zipped the state_dict on googledrive,
        # so would need to download, unzip and read (2.8gb file for a ~150mb state dict).
        # Would like to load dict from url but need somewhere to save the state dicts.
        raise NotImplementedError(
            "Currently not implemented. You need to manually download weights provided by the paper's author"
            " and load then to the model with `state_dict`. See https://github.com/Tencent/MedicalNet"
        )
    return model


def resnet10(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-10 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet10", ResNetBlock, [1, 1, 1, 1], get_inplanes(), pretrained, progress, **kwargs)


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", ResNetBlock, [2, 2, 2, 2], get_inplanes(), pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", ResNetBlock, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", ResNetBottleneck, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)


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

import math
import operator
import re
from functools import reduce
from typing import List, NamedTuple, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.utils import model_zoo

from monai.networks.layers.factories import Act, Conv, Pad, Pool
from monai.networks.layers.utils import get_norm_layer
from monai.utils.module import look_up_option

__all__ = [
    "EfficientNet",
    "EfficientNetBN",
    "get_efficientnet_image_size",
    "drop_connect",
    "EfficientNetBNFeatures",
    "BlockArgs",
]

efficientnet_params = {
    # model_name: (width_mult, depth_mult, image_size, dropout_rate, dropconnect_rate)
    "efficientnet-b0": (1.0, 1.0, 224, 0.2, 0.2),
    "efficientnet-b1": (1.0, 1.1, 240, 0.2, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3, 0.2),
    "efficientnet-b3": (1.2, 1.4, 300, 0.3, 0.2),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4, 0.2),
    "efficientnet-b5": (1.6, 2.2, 456, 0.4, 0.2),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5, 0.2),
    "efficientnet-b7": (2.0, 3.1, 600, 0.5, 0.2),
    "efficientnet-b8": (2.2, 3.6, 672, 0.5, 0.2),
    "efficientnet-l2": (4.3, 5.3, 800, 0.5, 0.2),
}

url_map = {
    "efficientnet-b0": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
    "efficientnet-b1": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
    "efficientnet-b2": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
    "efficientnet-b3": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
    "efficientnet-b4": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
    "efficientnet-b5": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
    "efficientnet-b6": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
    "efficientnet-b7": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth",
    # trained with adversarial examples, simplify the name to decrease string length
    "b0-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth",
    "b1-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth",
    "b2-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth",
    "b3-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth",
    "b4-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth",
    "b5-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth",
    "b6-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth",
    "b7-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth",
    "b8-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth",
}


class MBConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        image_size: List[int],
        expand_ratio: int,
        se_ratio: Optional[float],
        id_skip: Optional[bool] = True,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
        drop_connect_rate: Optional[float] = 0.2,
        se_module='se',
    ) -> None:
        """
        Mobile Inverted Residual Bottleneck Block.

        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: size of the kernel for conv ops.
            stride: stride to use for conv ops.
            image_size: input image resolution.
            expand_ratio: expansion ratio for inverted bottleneck.
            se_ratio: squeeze-excitation ratio for se layers.
            id_skip: whether to use skip connection.
            norm: feature normalization type and arguments. Defaults to batch norm.
            drop_connect_rate: dropconnect rate for drop connection (individual weights) layers.

        References:
            [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
            [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
            [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
        """
        super().__init__()

        # select the type of N-Dimensional layers to use
        # these are based on spatial dims and selected from MONAI factories
        conv_type = Conv["conv", spatial_dims]
        adaptivepool_type = Pool["adaptiveavg", spatial_dims]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.id_skip = id_skip
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.drop_connect_rate = drop_connect_rate
        self.se_module = se_module
        
        if (se_ratio is not None) and (0.0 < se_ratio <= 1.0):
            self.has_se = True
            self.se_ratio = se_ratio
        else:
            self.has_se = False

        # Expansion phase (Inverted Bottleneck)
        inp = in_channels  # number of input channels
        oup = in_channels * expand_ratio  # number of output channels
        if self.expand_ratio != 1:
            self._expand_conv = conv_type(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._expand_conv_padding = _make_same_padder(self._expand_conv, image_size)

            self._bn0 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=oup)
        else:
            # need to have the following to fix JIT error:
            #   "Module 'MBConvBlock' has no attribute '_expand_conv'"

            # FIXME: find a better way to bypass JIT error
            self._expand_conv = nn.Identity()
            self._expand_conv_padding = nn.Identity()
            self._bn0 = nn.Identity()

        # Depthwise convolution phase
        self._depthwise_conv = conv_type(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=kernel_size,
            stride=self.stride,
            bias=False,
        )
        self._depthwise_conv_padding = _make_same_padder(self._depthwise_conv, image_size)
        self._bn1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=oup)
        image_size = _calculate_output_image_size(image_size, self.stride)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            if se_module == 'se':
                self._se_adaptpool = adaptivepool_type(1)
                num_squeezed_channels = max(1, int(in_channels * self.se_ratio))
                self._se_reduce = conv_type(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
                self._se_reduce_padding = _make_same_padder(self._se_reduce, [1, 1])
                self._se_expand = conv_type(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
                self._se_expand_padding = _make_same_padder(self._se_expand, [1, 1])
            elif se_module =='acm':
                num_acm_groups = 4
                orthogonal_loss= False
                self.se = nets.ACM(num_heads=num_acm_groups, num_features=oup, orthogonal_loss=orthogonal_loss)
            elif se_module =='nlnn':
                self.se = NLBlockND(in_channels=oup, mode='embedded', dimension=spatial_dims, norm_layer=norm)
            elif se_module =='deeprft':
                self.se = FFT_ConvBlock(oup,oup)
            elif se_module =='cbam':
                self.se = BAM(gate_channels=oup, reduction_ratio=16, pool_types=['avg', 'max'])
                
        # Pointwise convolution phase
        final_oup = out_channels
        self._project_conv = conv_type(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._project_conv_padding = _make_same_padder(self._project_conv, image_size)
        self._bn2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=final_oup)

        # swish activation to use - using memory efficient swish by default
        # can be switched to normal swish using self.set_swish() function call
        self._swish = Act["memswish"](inplace=True)

    def forward(self, inputs: torch.Tensor):
        """MBConvBlock"s forward function.

        Args:
            inputs: Input tensor.

        Returns:
            Output of this block after processing.
        """
        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(self._expand_conv_padding(x))
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(self._depthwise_conv_padding(x))
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            if self.se_module =='se':
                x_squeezed = self._se_adaptpool(x)
                x_squeezed = self._se_reduce(self._se_reduce_padding(x_squeezed))
                x_squeezed = self._swish(x_squeezed)
                x_squeezed = self._se_expand(self._se_expand_padding(x_squeezed))
                x = torch.sigmoid(x_squeezed) * x
            else:
                x = self.se(x)

        # Pointwise Convolution
        x = self._project_conv(self._project_conv_padding(x))
        x = self._bn2(x)

        # Skip connection and drop connect
        if self.id_skip and self.stride == 1 and self.in_channels == self.out_channels:
            # the combination of skip connection and drop connect brings about stochastic depth.
            if self.drop_connect_rate:
                x = drop_connect(x, p=self.drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient: bool = True) -> None:
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = Act["memswish"](inplace=True) if memory_efficient else Act["swish"](alpha=1.0)


class EfficientNet(nn.Module):
    def __init__(
        self,
        blocks_args_str: List[str],
        spatial_dims: int = 2,
        in_channels: int = 3,
        num_classes: int = 1000,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        dropout_rate: float = 0.2,
        image_size: int = 224,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
        drop_connect_rate: float = 0.2,
        depth_divisor: int = 8,
        se_module='se'
    ) -> None:
        """
        EfficientNet based on `Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/pdf/1905.11946.pdf>`_.
        Adapted from `EfficientNet-PyTorch <https://github.com/lukemelas/EfficientNet-PyTorch>`_.

        Args:
            blocks_args_str: block definitions.
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            num_classes: number of output classes.
            width_coefficient: width multiplier coefficient (w in paper).
            depth_coefficient: depth multiplier coefficient (d in paper).
            dropout_rate: dropout rate for dropout layers.
            image_size: input image resolution.
            norm: feature normalization type and arguments.
            drop_connect_rate: dropconnect rate for drop connection (individual weights) layers.
            depth_divisor: depth divisor for channel rounding.

        """
        super().__init__()

        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims can only be 1, 2 or 3.")

        # select the type of N-Dimensional layers to use
        # these are based on spatial dims and selected from MONAI factories
        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv["conv", spatial_dims]
        adaptivepool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            "adaptiveavg", spatial_dims
        ]

        # decode blocks args into arguments for MBConvBlock
        blocks_args = [BlockArgs.from_string(s) for s in blocks_args_str]

        # checks for successful decoding of blocks_args_str
        if not isinstance(blocks_args, list):
            raise ValueError("blocks_args must be a list")

        if blocks_args == []:
            raise ValueError("block_args must be non-empty")

        self._blocks_args = blocks_args
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.drop_connect_rate = drop_connect_rate

        # expand input image dimensions to list
        current_image_size = [image_size] * spatial_dims

        # Stem
        stride = 2
        out_channels = _round_filters(32, width_coefficient, depth_divisor)  # number of output channels
        self._conv_stem = conv_type(self.in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self._conv_stem_padding = _make_same_padder(self._conv_stem, current_image_size)
        self._bn0 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels)
        current_image_size = _calculate_output_image_size(current_image_size, stride)

        # build MBConv blocks
        num_blocks = 0
        self._blocks = nn.Sequential()

        self.extract_stacks = []

        # update baseline blocks to input/output filters and number of repeats based on width and depth multipliers.
        for idx, block_args in enumerate(self._blocks_args):
            block_args = block_args._replace(
                input_filters=_round_filters(block_args.input_filters, width_coefficient, depth_divisor),
                output_filters=_round_filters(block_args.output_filters, width_coefficient, depth_divisor),
                num_repeat=_round_repeats(block_args.num_repeat, depth_coefficient),
            )
            self._blocks_args[idx] = block_args

            # calculate the total number of blocks - needed for drop_connect estimation
            num_blocks += block_args.num_repeat

            if block_args.stride > 1:
                self.extract_stacks.append(idx)

        self.extract_stacks.append(len(self._blocks_args))

        # create and add MBConvBlocks to self._blocks
        idx = 0  # block index counter
        for stack_idx, block_args in enumerate(self._blocks_args):
            blk_drop_connect_rate = self.drop_connect_rate

            # scale drop connect_rate
            if blk_drop_connect_rate:
                blk_drop_connect_rate *= float(idx) / num_blocks

            sub_stack = nn.Sequential()
            # the first block needs to take care of stride and filter size increase.
            sub_stack.add_module(
                str(idx),
                MBConvBlock(
                    spatial_dims=spatial_dims,
                    in_channels=block_args.input_filters,
                    out_channels=block_args.output_filters,
                    kernel_size=block_args.kernel_size,
                    stride=block_args.stride,
                    image_size=current_image_size,
                    expand_ratio=block_args.expand_ratio,
                    se_ratio=block_args.se_ratio,
                    id_skip=block_args.id_skip,
                    norm=norm,
                    drop_connect_rate=blk_drop_connect_rate,
                    se_module=se_module
                ),
            )
            idx += 1  # increment blocks index counter

            current_image_size = _calculate_output_image_size(current_image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)

            # add remaining block repeated num_repeat times
            for _ in range(block_args.num_repeat - 1):
                blk_drop_connect_rate = self.drop_connect_rate

                # scale drop connect_rate
                if blk_drop_connect_rate:
                    blk_drop_connect_rate *= float(idx) / num_blocks

                # add blocks
                sub_stack.add_module(
                    str(idx),
                    MBConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_args.input_filters,
                        out_channels=block_args.output_filters,
                        kernel_size=block_args.kernel_size,
                        stride=block_args.stride,
                        image_size=current_image_size,
                        expand_ratio=block_args.expand_ratio,
                        se_ratio=block_args.se_ratio,
                        id_skip=block_args.id_skip,
                        norm=norm,
                        drop_connect_rate=blk_drop_connect_rate,
                        se_module=se_module
                    ),
                )
                idx += 1  # increment blocks index counter

            self._blocks.add_module(str(stack_idx), sub_stack)

        # sanity check to see if len(self._blocks) equal expected num_blocks
        if idx != num_blocks:
            raise ValueError("total number of blocks created != num_blocks")

        # Head
        head_in_channels = block_args.output_filters
        out_channels = _round_filters(1280, width_coefficient, depth_divisor)
        self._conv_head = conv_type(head_in_channels, out_channels, kernel_size=1, bias=False)
        self._conv_head_padding = _make_same_padder(self._conv_head, current_image_size)
        self._bn1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels)

        # final linear layer
        self._avg_pooling = adaptivepool_type(1)
        self._dropout = nn.Dropout(dropout_rate)
        self._fc = nn.Linear(out_channels, self.num_classes)

        # swish activation to use - using memory efficient swish by default
        # can be switched to normal swish using self.set_swish() function call
        self._swish = Act["memswish"]()

        # initialize weights using Tensorflow's init method from official impl.
        self._initialize_weights()


    def set_swish(self, memory_efficient: bool = True) -> None:
        """
        Sets swish function as memory efficient (for training) or standard (for JIT export).

        Args:
            memory_efficient: whether to use memory-efficient version of swish.

        """
        self._swish = Act["memswish"]() if memory_efficient else Act["swish"](alpha=1.0)
        for sub_stack in self._blocks:
            for block in sub_stack:
                block.set_swish(memory_efficient)


    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input should have spatially N dimensions
            ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.

        Returns:
            a torch Tensor of classification prediction in shape ``(Batch, num_classes)``.
        """
        # Stem
        x = self._conv_stem(self._conv_stem_padding(inputs))
        x = self._swish(self._bn0(x))
        # Blocks
        x = self._blocks(x)
        # Head
        x = self._conv_head(self._conv_head_padding(x))
        x = self._swish(self._bn1(x))

        # Pooling and final linear layer
        x = self._avg_pooling(x)

        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x


    def _initialize_weights(self) -> None:
        """
        Args:
            None, initializes weights for conv/linear/batchnorm layers
            following weight init methods from
            `official Tensorflow EfficientNet implementation
            <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py#L61>`_.
            Adapted from `EfficientNet-PyTorch's init method
            <https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/efficientnet_builder.py>`_.
        """
        for _, m in self.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                fan_out = reduce(operator.mul, m.kernel_size, 1) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                fan_in = 0
                init_range = 1.0 / math.sqrt(fan_in + fan_out)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()



class EfficientNetBN(EfficientNet):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        progress: bool = True,
        spatial_dims: int = 2,
        in_channels: int = 3,
        num_classes: int = 1000,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
        adv_prop: bool = False,
    ) -> None:
        """
        Generic wrapper around EfficientNet, used to initialize EfficientNet-B0 to EfficientNet-B7 models
        model_name is mandatory argument as there is no EfficientNetBN itself,
        it needs the N in [0, 1, 2, 3, 4, 5, 6, 7, 8] to be a model

        Args:
            model_name: name of model to initialize, can be from [efficientnet-b0, ..., efficientnet-b8, efficientnet-l2].
            pretrained: whether to initialize pretrained ImageNet weights, only available for spatial_dims=2 and batch
                norm is used.
            progress: whether to show download progress for pretrained weights download.
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            num_classes: number of output classes.
            norm: feature normalization type and arguments.
            adv_prop: whether to use weights trained with adversarial examples.
                This argument only works when `pretrained` is `True`.

        Examples::

            # for pretrained spatial 2D ImageNet
            >>> image_size = get_efficientnet_image_size("efficientnet-b0")
            >>> inputs = torch.rand(1, 3, image_size, image_size)
            >>> model = EfficientNetBN("efficientnet-b0", pretrained=True)
            >>> model.eval()
            >>> outputs = model(inputs)

            # create spatial 2D
            >>> model = EfficientNetBN("efficientnet-b0", spatial_dims=2)

            # create spatial 3D
            >>> model = EfficientNetBN("efficientnet-b0", spatial_dims=3)

            # create EfficientNetB7 for spatial 2D
            >>> model = EfficientNetBN("efficientnet-b7", spatial_dims=2)

        """
        # block args
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ]

        # check if model_name is valid model
        if model_name not in efficientnet_params.keys():
            raise ValueError(
                "invalid model_name {} found, must be one of {} ".format(
                    model_name, ", ".join(efficientnet_params.keys())
                )
            )

        # get network parameters
        weight_coeff, depth_coeff, image_size, dropout_rate, dropconnect_rate = efficientnet_params[model_name]

        # create model and initialize random weights
        super().__init__(
            blocks_args_str=blocks_args_str,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            width_coefficient=weight_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            image_size=image_size,
            drop_connect_rate=dropconnect_rate,
            norm=norm,
        )

        # only pretrained for when `spatial_dims` is 2
        if pretrained and (spatial_dims == 2):
            _load_state_dict(self, model_name, progress, adv_prop)



class EfficientNetBNFeatures(EfficientNet):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        progress: bool = True,
        spatial_dims: int = 2,
        in_channels: int = 3,
        num_classes: int = 1000,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
        adv_prop: bool = False,
        se_module='se'
    ) -> None:
        """
        Initialize EfficientNet-B0 to EfficientNet-B7 models as a backbone, the backbone can
        be used as an encoder for segmentation and objection models.
        Compared with the class `EfficientNetBN`, the only different place is the forward function.

        This class refers to `PyTorch image models <https://github.com/rwightman/pytorch-image-models>`_.

        """
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ]

        # check if model_name is valid model
        if model_name not in efficientnet_params.keys():
            raise ValueError(
                "invalid model_name {} found, must be one of {} ".format(
                    model_name, ", ".join(efficientnet_params.keys())
                )
            )

        # get network parameters
        weight_coeff, depth_coeff, image_size, dropout_rate, dropconnect_rate = efficientnet_params[model_name]

        # create model and initialize random weights
        super().__init__(
            blocks_args_str=blocks_args_str,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            width_coefficient=weight_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            image_size=image_size,
            drop_connect_rate=dropconnect_rate,
            norm=norm,
            se_module=se_module
        )

        # only pretrained for when `spatial_dims` is 2
        if pretrained and (spatial_dims == 2):
            _load_state_dict(self, model_name, progress, adv_prop)


    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input should have spatially N dimensions
            ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.

        Returns:
            a list of torch Tensors.
        """
        # Stem
        x = self._conv_stem(self._conv_stem_padding(inputs))
        x = self._swish(self._bn0(x))

        features = []
        if 0 in self.extract_stacks:
            features.append(x)
        for i, block in enumerate(self._blocks):
            x = block(x)
            if i + 1 in self.extract_stacks:
                features.append(x)
        return features



def get_efficientnet_image_size(model_name: str) -> int:
    """
    Get the input image size for a given efficientnet model.

    Args:
        model_name: name of model to initialize, can be from [efficientnet-b0, ..., efficientnet-b7].

    Returns:
        Image size for single spatial dimension as integer.

    """
    # check if model_name is valid model
    if model_name not in efficientnet_params.keys():
        raise ValueError(
            "invalid model_name {} found, must be one of {} ".format(model_name, ", ".join(efficientnet_params.keys()))
        )

    # return input image size (all dims equal so only need to return for one dim)
    _, _, res, _, _ = efficientnet_params[model_name]
    return res


def drop_connect(inputs: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """
    Drop connect layer that drops individual connections.
    Differs from dropout as dropconnect drops connections instead of whole neurons as in dropout.

    Based on `Deep Networks with Stochastic Depth <https://arxiv.org/pdf/1603.09382.pdf>`_.
    Adapted from `Official Tensorflow EfficientNet utils
    <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/utils.py>`_.

    This function is generalized for MONAI's N-Dimensional spatial activations
    e.g. 1D activations [B, C, H], 2D activations [B, C, H, W] and 3D activations [B, C, H, W, D]

    Args:
        inputs: input tensor with [B, C, dim_1, dim_2, ..., dim_N] where N=spatial_dims.
        p: probability to use for dropping connections.
        training: whether in training or evaluation mode.

    Returns:
        output: output tensor after applying drop connection.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"p must be in range of [0, 1], found {p}")

    # eval mode: drop_connect is switched off - so return input without modifying
    if not training:
        return inputs

    # train mode: calculate and apply drop_connect
    batch_size: int = inputs.shape[0]
    keep_prob: float = 1 - p
    num_dims: int = len(inputs.shape) - 2

    # build dimensions for random tensor, use num_dims to populate appropriate spatial dims
    random_tensor_shape: List[int] = [batch_size, 1] + [1] * num_dims

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor: torch.Tensor = torch.rand(random_tensor_shape, dtype=inputs.dtype, device=inputs.device)
    random_tensor += keep_prob

    # round to form binary tensor
    binary_tensor: torch.Tensor = torch.floor(random_tensor)

    # drop connect using binary tensor
    output: torch.Tensor = inputs / keep_prob * binary_tensor
    return output


def _load_state_dict(model: nn.Module, arch: str, progress: bool, adv_prop: bool) -> None:
    if adv_prop:
        arch = arch.split("efficientnet-")[-1] + "-ap"
    model_url = look_up_option(arch, url_map, None)
    if model_url is None:
        print(f"pretrained weights of {arch} is not provided")
    else:
        # load state dict from url
        model_url = url_map[arch]
        pretrain_state_dict = model_zoo.load_url(model_url, progress=progress)
        model_state_dict = model.state_dict()

        pattern = re.compile(r"(.+)\.\d+(\.\d+\..+)")
        for key, value in model_state_dict.items():
            pretrain_key = re.sub(pattern, r"\1\2", key)
            if pretrain_key in pretrain_state_dict and value.shape == pretrain_state_dict[pretrain_key].shape:
                model_state_dict[key] = pretrain_state_dict[pretrain_key]

        model.load_state_dict(model_state_dict)


def _get_same_padding_conv_nd(
    image_size: List[int], kernel_size: Tuple[int, ...], dilation: Tuple[int, ...], stride: Tuple[int, ...]
) -> List[int]:
    """
    Helper for getting padding (nn.ConstantPadNd) to be used to get SAME padding
    conv operations similar to Tensorflow's SAME padding.

    This function is generalized for MONAI's N-Dimensional spatial operations (e.g. Conv1D, Conv2D, Conv3D)

    Args:
        image_size: input image/feature spatial size.
        kernel_size: conv kernel's spatial size.
        dilation: conv dilation rate for Atrous conv.
        stride: stride for conv operation.

    Returns:
        paddings for ConstantPadNd padder to be used on input tensor to conv op.
    """
    # get number of spatial dimensions, corresponds to kernel size length
    num_dims = len(kernel_size)

    # additional checks to populate dilation and stride (in case they are single entry tuples)
    if len(dilation) == 1:
        dilation = dilation * num_dims

    if len(stride) == 1:
        stride = stride * num_dims

    # equation to calculate (pad^+ + pad^-) size
    _pad_size: List[int] = [
        max((math.ceil(_i_s / _s) - 1) * _s + (_k_s - 1) * _d + 1 - _i_s, 0)
        for _i_s, _k_s, _d, _s in zip(image_size, kernel_size, dilation, stride)
    ]
    # distribute paddings into pad^+ and pad^- following Tensorflow's same padding strategy
    _paddings: List[Tuple[int, int]] = [(_p // 2, _p - _p // 2) for _p in _pad_size]

    # unroll list of tuples to tuples, and then to list
    # reversed as nn.ConstantPadNd expects paddings starting with last dimension
    _paddings_ret: List[int] = [outer for inner in reversed(_paddings) for outer in inner]
    return _paddings_ret


def _make_same_padder(conv_op: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], image_size: List[int]):
    """
    Helper for initializing ConstantPadNd with SAME padding similar to Tensorflow.
    Uses output of _get_same_padding_conv_nd() to get the padding size.

    This function is generalized for MONAI's N-Dimensional spatial operations (e.g. Conv1D, Conv2D, Conv3D)

    Args:
        conv_op: nn.ConvNd operation to extract parameters for op from
        image_size: input image/feature spatial size

    Returns:
        If padding required then nn.ConstandNd() padder initialized to paddings otherwise nn.Identity()
    """
    # calculate padding required
    padding: List[int] = _get_same_padding_conv_nd(image_size, conv_op.kernel_size, conv_op.dilation, conv_op.stride)

    # initialize and return padder
    padder = Pad["constantpad", len(padding) // 2]
    if sum(padding) > 0:
        return padder(padding=padding, value=0.0)
    return nn.Identity()


def _round_filters(filters: int, width_coefficient: Optional[float], depth_divisor: float) -> int:
    """
    Calculate and round number of filters based on width coefficient multiplier and depth divisor.

    Args:
        filters: number of input filters.
        width_coefficient: width coefficient for model.
        depth_divisor: depth divisor to use.

    Returns:
        new_filters: new number of filters after calculation.
    """

    if not width_coefficient:
        return filters

    multiplier: float = width_coefficient
    divisor: float = depth_divisor
    filters_float: float = filters * multiplier

    # follow the formula transferred from official TensorFlow implementation
    new_filters: float = max(divisor, int(filters_float + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters_float:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def _round_repeats(repeats: int, depth_coefficient: Optional[float]) -> int:
    """
    Re-calculate module's repeat number of a block based on depth coefficient multiplier.

    Args:
        repeats: number of original repeats.
        depth_coefficient: depth coefficient for model.

    Returns:
        new repeat: new number of repeat after calculating.
    """
    if not depth_coefficient:
        return repeats

    # follow the formula transferred from official TensorFlow impl.
    return int(math.ceil(depth_coefficient * repeats))


def _calculate_output_image_size(input_image_size: List[int], stride: Union[int, Tuple[int]]):
    """
    Calculates the output image size when using _make_same_padder with a stride.
    Required for static padding.

    Args:
        input_image_size: input image/feature spatial size.
        stride: Conv2d operation"s stride.

    Returns:
        output_image_size: output image/feature spatial size.
    """

    # checks to extract integer stride in case tuple was received
    if isinstance(stride, tuple):
        all_strides_equal = all(stride[0] == s for s in stride)
        if not all_strides_equal:
            raise ValueError(f"unequal strides are not possible, got {stride}")

        stride = stride[0]

    # return output image size
    return [int(math.ceil(im_sz / stride)) for im_sz in input_image_size]


class BlockArgs(NamedTuple):
    """
    BlockArgs object to assist in decoding string notation
        of arguments for MBConvBlock definition.
    """

    num_repeat: int
    kernel_size: int
    stride: int
    expand_ratio: int
    input_filters: int
    output_filters: int
    id_skip: bool
    se_ratio: Optional[float] = None

    @staticmethod
    def from_string(block_string: str):
        """
        Get a BlockArgs object from a string notation of arguments.

        Args:
            block_string (str): A string notation of arguments.
                                Examples: "r1_k3_s11_e1_i32_o16_se0.25".

        Returns:
            BlockArgs: namedtuple defined at the top of this function.
        """
        ops = block_string.split("_")
        options = {}
        for op in ops:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # check stride
        stride_check = (
            ("s" in options and len(options["s"]) == 1)
            or (len(options["s"]) == 2 and options["s"][0] == options["s"][1])
            or (len(options["s"]) == 3 and options["s"][0] == options["s"][1] and options["s"][0] == options["s"][2])
        )
        if not stride_check:
            raise ValueError("invalid stride option received")

        return BlockArgs(
            num_repeat=int(options["r"]),
            kernel_size=int(options["k"]),
            stride=int(options["s"][0]),
            expand_ratio=int(options["e"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
        )


    def to_string(self):
        """
        Return a block string notation for current BlockArgs object

        Returns:
            A string notation of BlockArgs object arguments.
                Example: "r1_k3_s11_e1_i32_o16_se0.25_noskip".
        """
        string = "r{}_k{}_s{}{}_e{}_i{}_o{}_se{}".format(
            self.num_repeat,
            self.kernel_size,
            self.stride,
            self.stride,
            self.expand_ratio,
            self.input_filters,
            self.output_filters,
            self.se_ratio,
        )

        if not self.id_skip:
            string += "_noskip"
        return string
    
class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
        # is_pad: bool = True,
        is_pad: bool = False,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.
            is_pad: whether to pad upsampling features to fit features from encoder. Defaults to True.
        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)
        self.is_pad = is_pad

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        """
        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            if self.is_pad:
                # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
                dimensions = len(x.shape) - 2
                sp = [0] * (dimensions * 2)
                for i in range(dimensions):
                    if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                        sp[i * 2 + 1] = 1
                # x_0 = torch.nn.functional.pad(x_0, sp, "constant")
                x_0 = torch.nn.functional.pad(x_0, sp, "replicate") # original
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x

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
#             if norm=='instance':
#                 self.encoder = bn2instance(self.encoder)
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

        
class UNet(nn.Module):
    def __init__(
        self,
        modelName='efficientnet-b3',
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 2,
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0, # (0.1, {"inplace": True}),
        upsample: str = "deconv", # [deconv, nontrainable, pixelshuffle]
        supervision = "NONE", #[None,'old','new']
        skipModule= "NONE",
        skipASPP = "NONE",
        se_module= 'se',
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Args:
            modelName: Backbone of efficientNet
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm. for group norm, norm=("group", {"num_groups": 4})
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            supervision : 'TYPE1,'TYPE2,'NONE'
            se_module : 'se', 'acm', 'ffc', 'deeprft', 'nlnn'
            skipModule : '[Module]_[BOTTOM#]'-->'FFC','DEEPRFT','ACM8','NONE','ACM2','ACM4','NLNN','SE'
                                             -->'BOTTOM5','BOTTOM4','BOTTOM3','BOTTOM2','BOTTOM1'
            skipASPP : -->'BOTTOM5','BOTTOM4','BOTTOM3','BOTTOM2','BOTTOM1','NONE'
            
        <example>
        net = UNet(modelName='efficientnet-b2', spatial_dims = 1, in_channels = 1, out_channels = 4, norm='instance', upsample='pixelshuffle', nnblock=True, ASPP='all', supervision=True, FFC='FFC', TRB=True)
        yhat = net(torch.rand(2,1,2048))

        """
        super().__init__()

        # U-net encoder
        if 'efficientnet' in modelName:
            ########################################################## preset init_ch
            # self.encoder = monai.networks.nets.EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=1, in_channels=1, norm=norm , num_classes=1000, adv_prop=True)
            self.encoder =EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=1, in_channels=1, norm=norm , num_classes=1000, adv_prop=True,se_module=se_module)
            x_test = torch.rand(2, 1, 2048)
            yhat_test = self.encoder(x_test)
            init_ch = yhat_test[0].shape[1]
            ########################################################## preset init_ch
            self.conv_0 = TwoConv(spatial_dims, in_channels, init_ch, act, norm, bias, dropout)
            # self.encoder = monai.networks.nets.EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=1, in_channels=init_ch, norm=norm , num_classes=1000, adv_prop=True)
            self.encoder = EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=1, in_channels=init_ch, norm=norm , num_classes=1000, adv_prop=True,se_module=se_module)
        elif 'resnet' in modelName:
            if modelName == 'resnet18':
                self.encoder = resnet18(spatial_dims=1, n_input_channels=1)
            elif modelName == 'resnet34':
                self.encoder = resnet34(spatial_dims=1, n_input_channels=1)
            elif modelName == 'resnet50':
                self.encoder = resnet50(spatial_dims=1, n_input_channels=1)
            x_test = torch.rand(2, 1, 2048)
            yhat_test = self.encoder(x_test)
            init_ch = yhat_test[0].shape[1]
            ########################################################## preset init_ch
            self.conv_0 = TwoConv(spatial_dims, in_channels, init_ch, act, norm, bias, dropout)
            if modelName == 'resnet18':
                self.encoder = nets.resnet18(spatial_dims=1, n_input_channels=init_ch)
            elif modelName == 'resnet34':
                self.encoder = nets.resnet34(spatial_dims=1, n_input_channels=init_ch)
            elif modelName == 'resnet50':
                self.encoder = nets.resnet50(spatial_dims=1, n_input_channels=init_ch)
        else:
            print('please check modelName')
        
        x = torch.rand(2, init_ch, 64)
        yhat = self.encoder(x)
        fea = [yhat_.shape[1] for yhat_ in yhat]
        print(fea)
        
        # skip modules
        self.skipModule = skipModule
        self.skip_1= None
        self.skip_2= None
        self.skip_3= None
        self.skip_4= None
        self.skip_5= None
        
        if skipModule=="NONE":
            print("skipModule:NONE")
            
        elif 'ACM' in skipModule:
            group = int(skipModule.split('ACM')[-1][0])
            self.ACMLambda = float(skipModule.split('_')[1])
            l = int(skipModule.split('BOTTOM')[-1])
            print(f"skipModule: {skipModule} group {group} ACMLambda {self.ACMLambda}")
            if self.ACMLambda==0:
                self.skip_1 = ACM(num_heads=fea[0]//group, num_features=fea[0], orthogonal_loss=False) if l>=5 else nn.Identity()
                self.skip_2 = ACM(num_heads=fea[1]//group, num_features=fea[1], orthogonal_loss=False) if l>=4 else nn.Identity()
                self.skip_3 = ACM(num_heads=fea[2]//group, num_features=fea[2], orthogonal_loss=False) if l>=3 else nn.Identity()
                self.skip_4 = ACM(num_heads=fea[3]//group, num_features=fea[3], orthogonal_loss=False) if l>=2 else nn.Identity()
                self.skip_5 = ACM(num_heads=fea[4]//group, num_features=fea[4], orthogonal_loss=False) if l>=1 else nn.Identity()
            else:
                self.skip_1 = ACM(num_heads=fea[0]//group, num_features=fea[0], orthogonal_loss=True) if l>=5 else nn.Identity()
                self.skip_2 = ACM(num_heads=fea[1]//group, num_features=fea[1], orthogonal_loss=True) if l>=4 else nn.Identity()
                self.skip_3 = ACM(num_heads=fea[2]//group, num_features=fea[2], orthogonal_loss=True) if l>=3 else nn.Identity()
                self.skip_4 = ACM(num_heads=fea[3]//group, num_features=fea[3], orthogonal_loss=True) if l>=2 else nn.Identity()
                self.skip_5 = ACM(num_heads=fea[4]//group, num_features=fea[4], orthogonal_loss=True) if l>=1 else nn.Identity()

#             self.skip_1 = nn.Sequential(nn.Conv1d(fea[0],(fea[0]//8)*32,1), ACM(num_heads=(fea[0]//8)*32//group, num_features=(fea[0]//8)*32, orthogonal_loss=False),nn.Conv1d((fea[0]//8)*32,fea[0],1),nn.InstanceNorm1d(fea[0])) 
#             self.skip_2 = nn.Sequential(nn.Conv1d(fea[1],(fea[1]//8)*32,1), ACM(num_heads=(fea[1]//8)*32//group, num_features=(fea[1]//8)*32, orthogonal_loss=False),nn.Conv1d((fea[1]//8)*32,fea[1],1),nn.InstanceNorm1d(fea[1]))
#             self.skip_3 = nn.Sequential(nn.Conv1d(fea[2],(fea[2]//8)*32,1), ACM(num_heads=(fea[2]//8)*32//group, num_features=(fea[2]//8)*32, orthogonal_loss=False),nn.Conv1d((fea[2]//8)*32,fea[2],1),nn.InstanceNorm1d(fea[2]))
#             self.skip_4 = nn.Sequential(nn.Conv1d(fea[3],(fea[3]//8)*32,1), ACM(num_heads=(fea[3]//8)*32//group, num_features=(fea[3]//8)*32, orthogonal_loss=False),nn.Conv1d((fea[3]//8)*32,fea[3],1),nn.InstanceNorm1d(fea[3]))
#             self.skip_5 = nn.Sequential(nn.Conv1d(fea[4],(fea[4]//8)*32,1), ACM(num_heads=(fea[4]//8)*32//group, num_features=(fea[4]//8)*32, orthogonal_loss=False),nn.Conv1d((fea[4]//8)*32,fea[4],1),nn.InstanceNorm1d(fea[4]))
            
        elif 'NLNN' in skipModule:
            norms = ['instance','batch']
            if not norm in norms:
                norm = None
            # norm = None
            l = int(skipModule.split('BOTTOM')[-1])
            print(f"skipModule: {skipModule} {l}")
            self.skip_1 = NLBlockND(in_channels=fea[0], mode='embedded', dimension=spatial_dims, norm_layer=norm) if l>=5 else nn.Identity()
            self.skip_2 = NLBlockND(in_channels=fea[1], mode='embedded', dimension=spatial_dims, norm_layer=norm) if l>=4 else nn.Identity()
            self.skip_3 = NLBlockND(in_channels=fea[2], mode='embedded', dimension=spatial_dims, norm_layer=norm) if l>=3 else nn.Identity()
            self.skip_4 = NLBlockND(in_channels=fea[3], mode='embedded', dimension=spatial_dims, norm_layer=norm) if l>=2 else nn.Identity()
            self.skip_5 = NLBlockND(in_channels=fea[4], mode='embedded', dimension=spatial_dims, norm_layer=norm) if l>=1 else nn.Identity()
        elif 'FFC' in skipModule:
            l = int(skipModule.split('BOTTOM')[-1])
            print(f"skipModule: {skipModule}")
            self.skip_1 = FFC_BN_ACT(fea[0],fea[0]) if l>=5 else nn.Identity()
            self.skip_2 = FFC_BN_ACT(fea[1],fea[1]) if l>=4 else nn.Identity()
            self.skip_3 = FFC_BN_ACT(fea[2],fea[2]) if l>=3 else nn.Identity()
            self.skip_4 = FFC_BN_ACT(fea[3],fea[3]) if l>=2 else nn.Identity()
            self.skip_5 = FFC_BN_ACT(fea[4],fea[4]) if l>=1 else nn.Identity()
        elif 'DEEPRFT' in skipModule:
            l = int(skipModule.split('BOTTOM')[-1])
            print(f"skipModule: {skipModule}")
            self.skip_1 = FFT_ConvBlock(fea[0],fea[0]) if l>=5 else nn.Identity()
            self.skip_2 = FFT_ConvBlock(fea[1],fea[1]) if l>=4 else nn.Identity()
            self.skip_3 = FFT_ConvBlock(fea[2],fea[2]) if l>=3 else nn.Identity()
            self.skip_4 = FFT_ConvBlock(fea[3],fea[3]) if l>=2 else nn.Identity()
            self.skip_5 = FFT_ConvBlock(fea[4],fea[4]) if l>=1 else nn.Identity()
        elif 'SE' in skipModule:
            l = int(skipModule.split('BOTTOM')[-1])
            print(f"skipModule: {skipModule}")
            self.skip_1 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[0]) if l>=5 else nn.Identity()
            self.skip_2 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[1]) if l>=4 else nn.Identity()
            self.skip_3 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[2]) if l>=3 else nn.Identity()
            self.skip_4 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[3]) if l>=2 else nn.Identity()
            self.skip_5 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[4]) if l>=1 else nn.Identity()
        elif 'CBAM' in skipModule:
            l = int(skipModule.split('BOTTOM')[-1])
            print(f"skipModule: {skipModule}")
            self.skip_1 = CBAM(gate_channels=fea[0], reduction_ratio=16, pool_types=['avg', 'max']) if l>=5 else nn.Identity()
            self.skip_2 = CBAM(gate_channels=fea[1], reduction_ratio=16, pool_types=['avg', 'max']) if l>=4 else nn.Identity()
            self.skip_3 = CBAM(gate_channels=fea[2], reduction_ratio=16, pool_types=['avg', 'max']) if l>=3 else nn.Identity()
            self.skip_4 = CBAM(gate_channels=fea[3], reduction_ratio=16, pool_types=['avg', 'max']) if l>=2 else nn.Identity()
            self.skip_5 = CBAM(gate_channels=fea[4], reduction_ratio=16, pool_types=['avg', 'max']) if l>=1 else nn.Identity()
            
        self.skipASPP=skipASPP
        self.ASPP_1 = nn.Identity()
        self.ASPP_2 = nn.Identity()
        self.ASPP_3 = nn.Identity()
        self.ASPP_4 = nn.Identity()
        self.ASPP_5 = nn.Identity()
        
        print(f"skipASPP:ASPP {self.skipASPP}")
        if self.skipASPP=='NONE':
            pass
        elif self.skipASPP=='BOTTOM1':
            self.ASPP_5 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[4], conv_out_channels=fea[4]//4, norm_type=norm, acti_type=act, bias=bias)
        elif self.skipASPP=='BOTTOM2':
            self.ASPP_4 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[3], conv_out_channels=fea[3]//4, norm_type=norm, acti_type=act, bias=bias)            
            self.ASPP_5 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[4], conv_out_channels=fea[4]//4, norm_type=norm, acti_type=act, bias=bias)
        elif self.skipASPP=='BOTTOM3':
            self.ASPP_3 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[2], conv_out_channels=fea[2]//4, norm_type=norm, acti_type=act, bias=bias)            
            self.ASPP_4 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[3], conv_out_channels=fea[3]//4, norm_type=norm, acti_type=act, bias=bias)            
            self.ASPP_5 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[4], conv_out_channels=fea[4]//4, norm_type=norm, acti_type=act, bias=bias)
        elif self.skipASPP=='BOTTOM4':
            self.ASPP_2 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[1], conv_out_channels=fea[1]//4, norm_type=norm, acti_type=act, bias=bias)            
            self.ASPP_3 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[2], conv_out_channels=fea[2]//4, norm_type=norm, acti_type=act, bias=bias)            
            self.ASPP_4 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[3], conv_out_channels=fea[3]//4, norm_type=norm, acti_type=act, bias=bias)            
            self.ASPP_5 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[4], conv_out_channels=fea[4]//4, norm_type=norm, acti_type=act, bias=bias)
        elif self.skipASPP=='BOTTOM5':
            self.ASPP_1 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[0], conv_out_channels=fea[0]//4, norm_type=norm, acti_type=act, bias=bias)            
            self.ASPP_2 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[1], conv_out_channels=fea[1]//4, norm_type=norm, acti_type=act, bias=bias)            
            self.ASPP_3 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[2], conv_out_channels=fea[2]//4, norm_type=norm, acti_type=act, bias=bias)            
            self.ASPP_4 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[3], conv_out_channels=fea[3]//4, norm_type=norm, acti_type=act, bias=bias)            
            self.ASPP_5 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[4], conv_out_channels=fea[4]//4, norm_type=norm, acti_type=act, bias=bias)
        
        # U-Net Decoder
        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample, interp_mode='linear')
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample, interp_mode='linear')
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample, interp_mode='linear')
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[0], act, norm, bias, dropout, upsample, interp_mode='linear')
        self.upcat_0 = UpCat(spatial_dims, fea[0], fea[0], fea[0], act, norm, bias, dropout, upsample, interp_mode='linear', halves=False)
        # self.final_conv = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
        self.final_conv = nn.Sequential(TwoConv(spatial_dims, fea[0], fea[0], act, norm, bias, dropout),
                                        Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1),)
        
        self.supervision = supervision
        if supervision=='old' or supervision =='TYPE1':
            # self.final_conv = Conv["conv", spatial_dims](fea[0]+fea[0]+fea[1]+fea[2]+fea[3]+fea[4], out_channels, kernel_size=1)
            self.final_conv = nn.Sequential(TwoConv(spatial_dims, fea[0]+fea[0]+fea[1]+fea[2]+fea[3]+fea[4], fea[0]+fea[0]+fea[1]+fea[2]+fea[3]+fea[4], act, norm, bias, dropout),
                                            Conv["conv", spatial_dims](fea[0]+fea[0]+fea[1]+fea[2]+fea[3]+fea[4], out_channels, kernel_size=1),)

            
        elif supervision=='new' or supervision =='TYPE2':
            self.sv0= Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=3, padding=1)
            self.sv1= Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=3, padding=1)
            self.sv2= Conv["conv", spatial_dims](fea[1], out_channels, kernel_size=3, padding=1)
            self.sv3= Conv["conv", spatial_dims](fea[2], out_channels, kernel_size=3, padding=1)
            self.sv4= Conv["conv", spatial_dims](fea[3], out_channels, kernel_size=3, padding=1)
            self.sv5= Conv["conv", spatial_dims](fea[4], out_channels, kernel_size=3, padding=1)
            # self.final_conv = Conv["conv", spatial_dims](out_channels*6, out_channels, kernel_size=1)
            self.final_conv = nn.Sequential(TwoConv(spatial_dims, out_channels*6, out_channels*6, act, norm, bias, dropout),
                                            Conv["conv", spatial_dims](out_channels*6, out_channels, kernel_size=1),)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        set_seed()
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
            neg_slope=1e-2
            module.weight = nn.init.kaiming_normal_(module.weight, a=neg_slope)
            if module.bias is not None:
                module.bias = torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.BatchNorm1d):
            if module.bias is not None:
                module.bias = torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        
        x0 = self.conv_0(x)
        x1, x2, x3, x4, x5 = self.encoder(x0)
        dp = False
        dp1 =dp2 = dp3 = dp4 = dp5 = 0.
        
        if self.skipModule=="NONE":
            pass
        else:
            x1 = self.skip_1(x1)
            x2 = self.skip_2(x2)
            x3 = self.skip_3(x3)
            x4 = self.skip_4(x4)
            x5 = self.skip_5(x5)
            
            if isinstance(x1,tuple) or isinstance(x1,list):
                x1, dp1 = x1
                dp1 = torch.abs(dp1.mean())
            if isinstance(x2,tuple) or isinstance(x2,list):
                x2, dp2 = x2 
                dp2 = torch.abs(dp2.mean())
            if isinstance(x3,tuple) or isinstance(x3,list):
                x3, dp3 = x3 
                dp3 = torch.abs(dp3.mean())
            if isinstance(x4,tuple) or isinstance(x4,list):
                x4, dp4 = x4 
                dp4 = torch.abs(dp4.mean())
            if isinstance(x5,tuple) or isinstance(x5,list):
                x5, dp5 = x5
                dp5 = torch.abs(dp5.mean())
            if dp1!=0 or dp2!=0 or dp!=0 or dp4!=0 or dp5!=0:
                dp = self.ACMLambda * (dp1+dp2+dp3+dp4+dp5)

            # x1 = x1 + self.skip_1(x1)
            # x2 = x2 + self.skip_2(x2)
            # x3 = x3 + self.skip_3(x3)
            # x4 = x4 + self.skip_4(x4)
            # x5 = x5 + self.skip_5(x5)
            
        if self.skipASPP==None or self.skipASPP=='NONE':
            pass
        else:
            x1 = self.ASPP_1(x1)
            x2 = self.ASPP_2(x2)
            x3 = self.ASPP_3(x3)
            x4 = self.ASPP_4(x4)
            x5 = self.ASPP_5(x5)

        u4 = self.upcat_4(x5, x4)
        u3 = self.upcat_3(u4, x3)
        u2 = self.upcat_2(u3, x2)
        u1 = self.upcat_1(u2, x1)
        u0 = self.upcat_0(u1, x0)
        # print(u0.shape, u1.shape, u2.shape, u3.shape, u4.shape)
        
        if self.supervision=='old' or self.supervision =='TYPE1':            
            s5 = _upsample_like(x5,u0)
            s4 = _upsample_like(x4,u0)
            s3 = _upsample_like(x3,u0)
            s2 = _upsample_like(x2,u0)
            s1 = _upsample_like(x1,u0)            
            u0 = torch.cat((u0,s1,s2,s3,s4,s5),1)            
            # print(u0.shape, s1.shape, s2.shape, s3.shape, s4.shape, s5.shape)
            
            logits = self.final_conv(u0)
            # print(logits.shape)
            # return torch.sigmoid(logits)
            return [torch.sigmoid(logits), dp] if dp else torch.sigmoid(logits)

        elif self.supervision=='new' or self.supervision =='TYPE2':
            s5 = self.sv5(_upsample_like(x5,u0))
            s4 = self.sv4(_upsample_like(x4,u0))
            s3 = self.sv3(_upsample_like(x3,u0))
            s2 = self.sv2(_upsample_like(x2,u0))
            s1 = self.sv1(_upsample_like(x1,u0))
            u0 = self.sv0(u0)
            # print(u0.shape, s1.shape, s2.shape, s3.shape, s4.shape, s5.shape)
            u0 = torch.cat((u0,s1,s2,s3,s4,s5),dim=1)
            
            logits = self.final_conv(u0)
            # print(logits.shape)
            # return torch.sigmoid(logits), torch.sigmoid(s1), torch.sigmoid(s2), torch.sigmoid(s3), torch.sigmoid(s4), torch.sigmoid(s5) 
            return [torch.sigmoid(logits), dp] if dp else torch.sigmoid(logits)
        
        else:
            logits = self.final_conv(u0)
            # print(logits.shape)
            return [torch.sigmoid(logits), dp] if dp else torch.sigmoid(logits)