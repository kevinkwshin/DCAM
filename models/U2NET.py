import torch
import torch.nn as nn
import torch.nn.functional as F

from .acm import *
from .cbam import *
from .deeprft import *
from .ffc import *
from .nnblock import *

import monai

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:],mode='linear')
    return src

class REBNCONV(nn.Module):
    def __init__(self,in_ch=1, out_ch=1, dirate=1, dropout=0.1, norm='instance'):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv1d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        if norm=='instance':
            self.bn_s1 = nn.InstanceNorm1d(out_ch)
        elif norm=='batch':
            self.bn_s1 = nn.BatchNorm1d(out_ch)    
        self.relu_s1 = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        xout = self.dropout(xout)
        
        return xout

### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, dropout=0.1, norm='instance'):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1,dropout=dropout,norm= norm)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1,dropout=dropout,norm= norm)
        self.pool1 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout,norm= norm)
        self.pool2 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout,norm= norm)
        self.pool3 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout,norm= norm)
        self.pool4 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout,norm= norm)
        self.pool5 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout,norm= norm)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2,dropout=dropout,norm= norm)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout,norm= norm)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout,norm= norm)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout,norm= norm)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout,norm= norm)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout,norm= norm)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1,dropout=dropout,norm= norm)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, dropout=0.1, norm='instance'):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1,dropout=dropout, norm=norm)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)
        self.pool1 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)
        self.pool2 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)
        self.pool3 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)
        self.pool4 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2,dropout=dropout, norm=norm)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm=norm)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm=norm)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm=norm)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm=norm)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1,dropout=dropout, norm=norm)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,dropout=0.1,norm='instance'):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1,dropout=dropout, norm= norm)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1,dropout=dropout, norm= norm)
        self.pool1 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm= norm)
        self.pool2 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm= norm)
        self.pool3 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm= norm)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2,dropout=dropout, norm= norm)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm= norm)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm= norm)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm= norm)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1,dropout=dropout, norm= norm)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,dropout=0.1,norm='instance'):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1,dropout=dropout, norm=norm)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)
        self.pool1 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)
        self.pool2 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1,dropout=dropout, norm=norm)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2,dropout=dropout, norm=norm)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm=norm)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1,dropout=dropout, norm=norm)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1,dropout=dropout, norm=norm)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,dropout=0.1, norm='instance'):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1,dropout=dropout, norm= norm)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1,dropout=dropout, norm= norm)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2,dropout=dropout, norm= norm)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4,dropout=dropout, norm= norm)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8,dropout=dropout, norm= norm)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4,dropout=dropout, norm= norm)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2,dropout=dropout, norm= norm)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1,dropout=dropout, norm= norm)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, encModule='NONE', decModule='NONE', temperature=1, dropout=0.1, norm='instance'):
        super(U2NET,self).__init__()

        self.stage1 = RSU7(in_ch,32,64,dropout=dropout, norm= norm)
        self.pool12 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128,dropout=dropout, norm= norm)
        self.pool23 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256,dropout=dropout, norm= norm)
        self.pool34 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512,dropout=dropout, norm= norm)
        self.pool45 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512,dropout=dropout, norm= norm)
        self.pool56 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512,dropout=dropout, norm= norm)

        # decoder
        self.stage5d = RSU4F(1024,256,512,dropout=dropout, norm= norm)
        self.stage4d = RSU4(1024,128,256,dropout=dropout, norm= norm)
        self.stage3d = RSU5(512,64,128,dropout=dropout, norm= norm)
        self.stage2d = RSU6(256,32,64,dropout=dropout, norm= norm)
        self.stage1d = RSU7(128,16,64,dropout=dropout, norm= norm)

        self.side1 = nn.Conv1d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv1d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv1d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv1d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv1d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv1d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv1d(6*out_ch,out_ch,1)
        
        fea = [64, 128, 256, 512, 512, 512]

        self.encModule = encModule
        print('U2-NET encModule is', encModule)

        self.encModule1 = nn.Identity()
        self.encModule2 = nn.Identity()
        self.encModule3 = nn.Identity()
        self.encModule4 = nn.Identity()
        self.encModule5 = nn.Identity()
        self.encModule6 = nn.Identity()

        if 'SE' in self.encModule:
            spatial_dims = 1
            self.encModule1 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[0])
            self.encModule2 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[1])
            self.encModule3 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[2])
            self.encModule4 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[3])
            self.encModule5 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[4])
            self.encModule6 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[5])              
        
        elif 'NN' in self.encModule:
            spatial_dims = 1
            self.encModule1 = NLBlockND(in_channels=fea[0], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.encModule2 = NLBlockND(in_channels=fea[1], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.encModule3 = NLBlockND(in_channels=fea[2], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.encModule4 = NLBlockND(in_channels=fea[3], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.encModule5 = NLBlockND(in_channels=fea[4], mode='embedded', dimension=spatial_dims, norm_layer=norm)     
            self.encModule6 = NLBlockND(in_channels=fea[5], mode='embedded', dimension=spatial_dims, norm_layer=norm)                      

        elif 'FFC' in self.encModule:
            self.encModule1 = FFC_BN_ACT(fea[0],fea[0])
            self.encModule2 = FFC_BN_ACT(fea[1],fea[1])
            self.encModule3 = FFC_BN_ACT(fea[2],fea[2])
            self.encModule4 = FFC_BN_ACT(fea[3],fea[3])
            self.encModule5 = FFC_BN_ACT(fea[4],fea[4])
            self.encModule6 = FFC_BN_ACT(fea[5],fea[5])            

        elif 'DEEPRFT' in self.encModule:
            self.encModule1 = FFT_ConvBlock(fea[0],fea[0])
            self.encModule2 = FFT_ConvBlock(fea[1],fea[1])
            self.encModule3 = FFT_ConvBlock(fea[2],fea[2])
            self.encModule4 = FFT_ConvBlock(fea[3],fea[3])
            self.encModule5 = FFT_ConvBlock(fea[4],fea[4])
            self.encModule6 = FFT_ConvBlock(fea[5],fea[5])
            
        elif 'ACM' in self.encModule:
            # group = 4
            # self.encModule1 = ACM(num_heads=fea[0]//group, num_features=fea[0], orthogonal_loss=False)
            # self.encModule2 = ACM(num_heads=fea[1]//group, num_features=fea[1], orthogonal_loss=False)
            # self.encModule3 = ACM(num_heads=fea[2]//group, num_features=fea[2], orthogonal_loss=False)
            # self.encModule4 = ACM(num_heads=fea[3]//group, num_features=fea[3], orthogonal_loss=False)
            # self.encModule5 = ACM(num_heads=fea[4]//group, num_features=fea[4], orthogonal_loss=False)
            # self.encModule6 = ACM(num_heads=fea[5]//group, num_features=fea[5], orthogonal_loss=False)

            group = 32
            self.decModule1 = ACM(num_heads=group, num_features=fea[0], orthogonal_loss=False)
            self.decModule2 = ACM(num_heads=group, num_features=fea[1], orthogonal_loss=False)
            self.decModule3 = ACM(num_heads=group, num_features=fea[2], orthogonal_loss=False)
            self.decModule4 = ACM(num_heads=group, num_features=fea[3], orthogonal_loss=False)
            self.decModule5 = ACM(num_heads=group, num_features=fea[4], orthogonal_loss=False)
            self.decModule6 = ACM(num_heads=group, num_features=fea[5], orthogonal_loss=False)

        elif 'MHA' in self.encModule:
            featureLength = 1280
            self.encModule1 = nn.MultiheadAttention(featureLength//1, 8, batch_first=True, dropout=0.01) 
            self.encModule2 = nn.MultiheadAttention(featureLength//2, 8, batch_first=True, dropout=0.01)
            self.encModule3 = nn.MultiheadAttention(featureLength//4, 8, batch_first=True, dropout=0.01)
            self.encModule4 = nn.MultiheadAttention(featureLength//8, 8, batch_first=True, dropout=0.01)
            self.encModule5 = nn.MultiheadAttention(featureLength//16, 8, batch_first=True, dropout=0.01)
            self.encModule6 = nn.MultiheadAttention(featureLength//32, 8, batch_first=True, dropout=0.01)
        
        self.decModule = decModule
        print('U2-NET decModule is', decModule)

        self.decModule1 = nn.Identity()
        self.decModule2 = nn.Identity()
        self.decModule3 = nn.Identity()
        self.decModule4 = nn.Identity()
        self.decModule5 = nn.Identity()
        self.decModule6 = nn.Identity()

        fea = [64, 64, 128, 256, 512, 512]

        if 'SE' in self.decModule:
            spatial_dims = 1
            self.decModule1 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[0]) # (128x256 and 512x256)
            self.decModule2 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[1])
            self.decModule3 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[2])
            self.decModule4 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[3])
            self.decModule5 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[4])
            self.decModule6 = monai.networks.blocks.ChannelSELayer(spatial_dims,fea[5])              

        elif 'NN' in self.decModule:
            spatial_dims = 1
            self.decModule1 = NLBlockND(in_channels=fea[0], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.decModule2 = NLBlockND(in_channels=fea[1], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.decModule3 = NLBlockND(in_channels=fea[2], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.decModule4 = NLBlockND(in_channels=fea[3], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.decModule5 = NLBlockND(in_channels=fea[4], mode='embedded', dimension=spatial_dims, norm_layer=norm)     
            self.decModule6 = NLBlockND(in_channels=fea[5], mode='embedded', dimension=spatial_dims, norm_layer=norm)                      

        elif 'FFC' in self.decModule:
            self.decModule1 = FFC_BN_ACT(fea[0],fea[0])
            self.decModule2 = FFC_BN_ACT(fea[1],fea[1])
            self.decModule3 = FFC_BN_ACT(fea[2],fea[2])
            self.decModule4 = FFC_BN_ACT(fea[3],fea[3])
            self.decModule5 = FFC_BN_ACT(fea[4],fea[4])
            self.decModule6 = FFC_BN_ACT(fea[5],fea[5])            

        elif 'DEEPRFT' in self.decModule:
            self.decModule1 = FFT_ConvBlock(fea[0],fea[0])
            self.decModule2 = FFT_ConvBlock(fea[1],fea[1])
            self.decModule3 = FFT_ConvBlock(fea[2],fea[2])
            self.decModule4 = FFT_ConvBlock(fea[3],fea[3])
            self.decModule5 = FFT_ConvBlock(fea[4],fea[4])
            self.decModule6 = FFT_ConvBlock(fea[5],fea[5])
            
        elif 'ACM' in self.decModule:
            # group = 4
            # self.decModule1 = ACM(num_heads=fea[0]//group, num_features=fea[0], orthogonal_loss=False)
            # self.decModule2 = ACM(num_heads=fea[1]//group, num_features=fea[1], orthogonal_loss=False)
            # self.decModule3 = ACM(num_heads=fea[2]//group, num_features=fea[2], orthogonal_loss=False)
            # self.decModule4 = ACM(num_heads=fea[3]//group, num_features=fea[3], orthogonal_loss=False)
            # self.decModule5 = ACM(num_heads=fea[4]//group, num_features=fea[4], orthogonal_loss=False)
            # self.decModule6 = ACM(num_heads=fea[5]//group, num_features=fea[5], orthogonal_loss=False)

            group = 32
            self.decModule1 = ACM(num_heads=group, num_features=fea[0], orthogonal_loss=False)
            self.decModule2 = ACM(num_heads=group, num_features=fea[1], orthogonal_loss=False)
            self.decModule3 = ACM(num_heads=group, num_features=fea[2], orthogonal_loss=False)
            self.decModule4 = ACM(num_heads=group, num_features=fea[3], orthogonal_loss=False)
            self.decModule5 = ACM(num_heads=group, num_features=fea[4], orthogonal_loss=False)
            self.decModule6 = ACM(num_heads=group, num_features=fea[5], orthogonal_loss=False)

        elif 'MHA' in self.decModule:
            featureLength = 1280
            self.decModule1 = nn.MultiheadAttention(featureLength//1, 8, batch_first=True, dropout=0.01) 
            self.decModule2 = nn.MultiheadAttention(featureLength//2, 8, batch_first=True, dropout=0.01)
            self.decModule3 = nn.MultiheadAttention(featureLength//4, 8, batch_first=True, dropout=0.01)
            self.decModule4 = nn.MultiheadAttention(featureLength//8, 8, batch_first=True, dropout=0.01)
            self.decModule5 = nn.MultiheadAttention(featureLength//16, 8, batch_first=True, dropout=0.01)
            self.decModule6 = nn.MultiheadAttention(featureLength//32, 8, batch_first=True, dropout=0.01)
 
        # featureLength = 1280
        # self.lastSelfAttention = nn.Identity()
        # self.lastSelfAttention = nn.MultiheadAttention(featureLength//1, 1, batch_first=True, dropout=0.01) 
        spatial_dims = 1
        self.lastSelfAttention = monai.networks.blocks.ChannelSELayer(spatial_dims, 12) 
        self.temperature = temperature
            
    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        if "MHA" not in self.encModule:    
           hx1 = self.encModule1(hx1)
        else: 
            hx1,_ = self.encModule1(hx1,hx1,hx1)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        if "MHA" not in self.encModule:    
            hx2 = self.encModule2(hx2)
        else: 
            hx2,_ = self.encModule2(hx2,hx2,hx2)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        if "MHA" not in self.encModule:    
            hx3 = self.encModule3(hx3)
        else: 
            hx3,_ = self.encModule3(hx3,hx3,hx3)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        if "MHA" not in self.encModule:    
            hx4 = self.encModule4(hx4)
        else: 
            hx4,_ = self.encModule4(hx4,hx4,hx4)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        if "MHA" not in self.encModule:    
            hx5 = self.encModule5(hx5)
        else: 
            hx5,_ = self.encModule5(hx5,hx5,hx5)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        if "MHA" not in self.encModule:    
            hx6 = self.encModule6(hx6)
        else: 
            hx6,_ = self.encModule6(hx6,hx6,hx6)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        if "MHA" not in self.decModule:    
            # print('hx5d', hx5d.shape)
            hx5d = self.decModule5(hx5d)
        else: 
            hx5d,_ = self.decModule5(hx5d,hx5d,hx5d)
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        if "MHA" not in self.decModule:    
            # print('hx4d',hx4d.shape)
            hx4d = self.decModule4(hx4d)
        else: 
            hx4d,_ = self.decModule4(hx4d,hx4d,hx4d)
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        if "MHA" not in self.decModule:    
            # print('hx3d',hx3d.shape)
            hx3d = self.decModule3(hx3d)
        else: 
            hx3d,_ = self.decModule3(hx3d,hx3d,hx3d)
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        if "MHA" not in self.decModule:    
            # print('hx2d',hx2d.shape)
            hx2d = self.decModule2(hx2d)
        else: 
            hx2d,_ = self.decModule2(hx2d,hx2d,hx2d)
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        if "MHA" not in self.decModule:    
            # print('hx1d',hx1d.shape)
            hx1d = self.decModule1(hx1d)
        else: 
            hx1d,_ = self.decModule1(hx1d,hx1d,hx1d)

        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = torch.cat((d1,d2,d3,d4,d5,d6),1)
        d0 = self.lastSelfAttention(d0)
        # d0,_ = self.lastSelfAttention(d0,d0,d0)
        d0 = self.outconv(d0)

        return torch.sigmoid(d0/self.temperature), torch.sigmoid(d1/self.temperature), torch.sigmoid(d2/self.temperature), torch.sigmoid(d3/self.temperature), torch.sigmoid(d4/self.temperature), torch.sigmoid(d5/self.temperature), torch.sigmoid(d6/self.temperature)

### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NETP,self).__init__()

        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool1d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(64,16,64)

        # decoder
        self.stage5d = RSU4F(128,16,64)
        self.stage4d = RSU4(128,16,64)
        self.stage3d = RSU5(128,16,64)
        self.stage2d = RSU6(128,16,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv1d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv1d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv1d(64,out_ch,3,padding=1)
        self.side4 = nn.Conv1d(64,out_ch,3,padding=1)
        self.side5 = nn.Conv1d(64,out_ch,3,padding=1)
        self.side6 = nn.Conv1d(64,out_ch,3,padding=1)

        self.outconv = nn.Conv1d(6*out_ch,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

bce_loss = nn.BCELoss()

def U2NETLOSS(yhat, y):
    d0, d1, d2, d3, d4, d5, d6 = yhat
    labels_v = y
    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss