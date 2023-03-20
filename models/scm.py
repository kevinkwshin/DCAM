import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, num_heads, num_features, scm_type=4):
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
    
