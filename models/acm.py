import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ACM(nn.Module):
    """
    if __name__ == '__main__':
        x1 = torch.randn(256 * 20 * 20 * 5).view(5, 256, 20, 20).float()
        x1 = torch.rand(2, 320, 160).float()
        acm = ACM(num_heads=32, num_features=320, orthogonal_loss=True)
        acm.init_parameters()
        y, dp = acm(x1)
        print(y.shape)
        print(dp.shape)

        ACM without orthogonal loss
        acm = ACM(num_heads=32, num_features=320, orthogonal_loss=False)
        acm.init_parameters()
        y = acm(x1)
        print(x1.shape,y.shape)
    """
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

