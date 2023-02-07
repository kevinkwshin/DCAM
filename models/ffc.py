
# 2d
import torch
import torch.nn as nn

class FFCSE_block(nn.Module):
    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=1, bias=True)
        self.relu1 = nn.LeakyReLU(0.1, inplace=False)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(channels // r, in_cg, kernel_size=1, bias=True)
        seltorch.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * seltorch.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * seltorch.sigmoid(self.conv_a2g(x))
        return x_l, x_g

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FourierUnit, self).__init__()
        
        self.conv_fft = torch.nn.Conv2d(in_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0)
        self.norm_fft   = nn.InstanceNorm2d(out_channels*2)
        self.relu   = nn.LeakyReLU(0.1)

    def forward(self, x):
        
        x_f = torch.fft.rfft2(x, norm='ortho')
        x_f = torch.cat([x_f.real, x_f.imag], dim=1)
        x_f = self.relu(self.norm_fft(self.conv_fft(x_f)))
                
        x_real, x_imag = torch.chunk(x_f, 2, dim=1)
        x_f = torch.complex(x_real, x_imag)
        x_f = torch.fft.irfft2(x_f, norm='ortho')
        
        return x_f

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()
            print(xs.shape)
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        return output

class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        
        self.in_cg = in_cg
        self.in_cl = in_cl
        self.out_cg = out_cg
        self.out_cl = out_cl
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)
        # print(self.convg2g)
        
    def forward(self, x):
        x_l, x_g = x[:,:self.in_cl], x[:,self.in_cl:]
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return torch.cat([out_xl, out_xg],dim=1)

class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.InstanceNorm2d, activation_layer=nn.Identity, enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias, enable_lfu)
        
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        
        self.in_cg = in_cg
        self.in_cl = in_cl
        self.out_cg = out_cg
        self.out_cl = out_cl       
        
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x = self.ffc(x)        
        x_l, x_g = x[:,:self.in_cl], x[:,self.in_cl:]
        # print(x_l.shape,x_g.shape)
        
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        # return x_l, x_g
        output = torch.cat([x_l,x_g],dim=1)
        return output

# 1d
import torch
import torch.nn as nn

class FFCSE_block(nn.Module):
    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool1d((1, 1))
        self.conv1 = nn.Conv1d(channels, channels // r, kernel_size=1, bias=True)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv1d(channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv1d(channels // r, in_cg, kernel_size=1, bias=True)
        seltorch.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * seltorch.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * seltorch.sigmoid(self.conv_a2g(x))
        return x_l, x_g

# 1d
class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FourierUnit, self).__init__()
        
        self.conv_fft = torch.nn.Conv1d(in_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0)
        self.norm_fft   = nn.InstanceNorm1d(out_channels*2)
        self.relu   = nn.LeakyReLU(0.1)

    def forward(self, x):
        
        x_f = torch.fft.rfft(x, norm='ortho')
        x_f = torch.cat([x_f.real, x_f.imag], dim=1)
        x_f = self.relu(self.norm_fft(self.conv_fft(x_f)))
        
        x_real, x_imag = torch.chunk(x_f, 2, dim=1)
        x_f = torch.complex(x_real, x_imag)
        x_f = torch.fft.irfft(x_f, norm='ortho')
        
        return x_f

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool1d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.InstanceNorm1d(out_channels // 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv1d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h = x.shape
            split_no = 2
            split_s_h = h // split_no
            # split_s_w = w // split_no
            xs = torch.cat(torch.split(x[:, :c // 2], split_s_h, dim=-1), dim=1).contiguous()
            # xs = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no).contiguous()
        else:
            xs = 0
        output = self.conv2(x + output + xs)
        return output

class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin=.5, ratio_gout=.5, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        
        super(FFC, self).__init__()
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        
        self.in_cg = in_cg
        self.in_cl = in_cl
        self.out_cg = out_cg
        self.out_cl = out_cl
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv1d
        self.convl2l = module(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv1d
        self.convl2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv1d
        self.convg2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)
        
    def forward(self, x):
        x_l, x_g = x[:,:self.in_cl], x[:,self.in_cl:]
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return torch.cat([out_xl, out_xg],dim=1)

class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio_gin=.5, ratio_gout=.5, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.InstanceNorm1d, activation_layer=nn.Identity, enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias, enable_lfu)
        
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        
        self.in_cg = in_cg
        self.in_cl = in_cl
        self.out_cg = out_cg
        self.out_cl = out_cl       
        
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x = self.ffc(x)        
        x_l, x_g = x[:,:self.in_cl], x[:,self.in_cl:]
        # print(x_l.shape,x_g.shape)
        
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        # return x_l, x_g
        output = torch.cat([x_l,x_g],dim=1)
        return output