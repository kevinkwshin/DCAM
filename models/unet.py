import torch
import torch.nn as nn
import torch.nn.functional as F

from .acm import *
from .cbam import *
from .deeprft import *
from .ffc import *
from .nnblock import *
from .resnet import *

import monai

import math
import operator
import re
from functools import reduce
from typing import List, NamedTuple, Optional, Tuple, Type, Union

import torch
from torch import nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.nets.basic_unet import TwoConv, Down, UpSample, Union
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Act, Conv, Pad, Pool

from monai.utils import deprecated_arg, ensure_tuple_rep

def _upsample_like(src,tar):
    # src = F.upsample(src,size=tar.shape[2:], mode='linear') # original
    src = F.upsample(src,size=tar.shape[2:], mode='nearest') # to deterministic function
    return src

class UNet(nn.Module):
    def __init__(
        self,
        modelName='efficientnet-b0',
        spatial_dims: int = 1,
        in_channels: int = 1,
        out_channels: int = 2,
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0, # (0.1, {"inplace": True}),
        upsample: str = "deconv", # [deconv, nontrainable, pixelshuffle]
        supervision = "NONE", #[None,'old','new']
        encModule="NONE",
        decModule="NONE",
        segheadModule ="NONE",
        featureLength = 1280,
        mtl="NONE"
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
            encModule : '[Module]_[BOTTOM#]'-->'FFC','DEEPRFT','ACM8','NONE','ACM2','ACM4','NLNN','SE'
                                             -->'BOTTOM5','BOTTOM4','BOTTOM3','BOTTOM2','BOTTOM1'
            
        <example>
        net = UNet(modelName='efficientnet-b2', spatial_dims = 1, in_channels = 1, out_channels = 4, norm='instance', upsample='pixelshuffle', nnblock=True, ASPP='all', supervision=True, FFC='FFC', TRB=True)
        yhat = net(torch.rand(2,1,2048))

        """
        super().__init__()

        # U-net encoder
        print(f'UNET encoder is {modelName}')
        if 'efficientnet' in modelName:
            ########################################################## preset init_ch
            self.encoder = monai.networks.nets.EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=spatial_dims, in_channels=1, norm=norm , num_classes=1000, adv_prop=True)
            # self.encoder =EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=1, in_channels=1, norm=norm , num_classes=1000, adv_prop=True)
            x_test = torch.rand(2, 1, featureLength)
            yhat_test = self.encoder(x_test)
            init_ch = yhat_test[0].shape[1]
            ########################################################## preset init_ch
            self.conv_0 = TwoConv(spatial_dims, in_channels, init_ch, act, norm, bias, dropout)
            self.encoder = monai.networks.nets.EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=spatial_dims, in_channels=init_ch, norm=norm , num_classes=1000, adv_prop=True)
            # self.encoder = EfficientNetBNFeatures(modelName, pretrained=True, progress=True, spatial_dims=1, in_channels=init_ch, norm=norm , num_classes=1000, adv_prop=True)

        elif 'resnet' in modelName:
            if modelName == 'resnet18':
                self.encoder = resnet18(spatial_dims=spatial_dims, n_input_channels=in_channels, norm=norm)
            elif modelName == 'resnet34':
                self.encoder = resnet34(spatial_dims=spatial_dims, n_input_channels=in_channels, norm=norm)
            elif modelName == 'resnet50':
                self.encoder = resnet50(spatial_dims=spatial_dims, n_input_channels=in_channels, norm=norm)

            x_test = torch.rand(2, in_channels, featureLength)
            yhat_test = self.encoder(x_test)
            init_ch = yhat_test[0].shape[1]
            ########################################################## preset init_ch
            self.conv_0 = TwoConv(spatial_dims, in_channels, init_ch, act, norm, bias, dropout)
            if modelName == 'resnet18':
                self.encoder = resnet18(spatial_dims=spatial_dims, n_input_channels=init_ch, norm=norm)
            elif modelName == 'resnet34':
                self.encoder = resnet34(spatial_dims=spatial_dims, n_input_channels=init_ch, norm=norm)
            elif modelName == 'resnet50':
                self.encoder = resnet50(spatial_dims=spatial_dims, n_input_channels=init_ch, norm=norm)

        else:
            print('please check modelName')
        
        x = torch.rand(2, init_ch, featureLength)
        yhat = self.encoder(x)
        fea = [yhat_.shape[1] for yhat_ in yhat]
        print(fea)
        
        # skip modules
        self.encModule = encModule
        self.encModule1 = nn.Identity()
        self.encModule2 = nn.Identity()
        self.encModule3 = nn.Identity()
        self.encModule4 = nn.Identity()
        self.encModule5 = nn.Identity()
        print(f"U-NET encModule is {encModule}")

        if 'ACM' in encModule:
            group = 4
            self.ACMLambda = 0.01
            if self.ACMLambda==0:
                self.encModule1 = ACM(num_heads=fea[0]//group, num_features=fea[0], orthogonal_loss=False)
                self.encModule2 = ACM(num_heads=fea[1]//group, num_features=fea[1], orthogonal_loss=False)
                self.encModule3 = ACM(num_heads=fea[2]//group, num_features=fea[2], orthogonal_loss=False)
                self.encModule4 = ACM(num_heads=fea[3]//group, num_features=fea[3], orthogonal_loss=False)
                self.encModule5 = ACM(num_heads=fea[4]//group, num_features=fea[4], orthogonal_loss=False)
            else:
                self.encModule1 = ACM(num_heads=fea[0]//group, num_features=fea[0], orthogonal_loss=True)
                self.encModule2 = ACM(num_heads=fea[1]//group, num_features=fea[1], orthogonal_loss=True)
                self.encModule3 = ACM(num_heads=fea[2]//group, num_features=fea[2], orthogonal_loss=True)
                self.encModule4 = ACM(num_heads=fea[3]//group, num_features=fea[3], orthogonal_loss=True)
                self.encModule5 = ACM(num_heads=fea[4]//group, num_features=fea[4], orthogonal_loss=True)
        elif 'NLNN' in encModule:
            norms = ['instance','batch']
            if not norm in norms:
                norm = None
            self.encModule1 = NLBlockND(in_channels=fea[0], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.encModule2 = NLBlockND(in_channels=fea[1], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.encModule3 = NLBlockND(in_channels=fea[2], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.encModule4 = NLBlockND(in_channels=fea[3], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.encModule5 = NLBlockND(in_channels=fea[4], mode='embedded', dimension=spatial_dims, norm_layer=norm)
        elif 'FFC' in encModule:
            self.encModule1 = FFC_BN_ACT(fea[0],fea[0])
            self.encModule2 = FFC_BN_ACT(fea[1],fea[1])
            self.encModule3 = FFC_BN_ACT(fea[2],fea[2])
            self.encModule4 = FFC_BN_ACT(fea[3],fea[3])
            self.encModule5 = FFC_BN_ACT(fea[4],fea[4])
        elif 'DEEPRFT' in encModule:
            self.encModule1 = FFT_ConvBlock(fea[0],fea[0])
            self.encModule2 = FFT_ConvBlock(fea[1],fea[1])
            self.encModule3 = FFT_ConvBlock(fea[2],fea[2])
            self.encModule4 = FFT_ConvBlock(fea[3],fea[3])
            self.encModule5 = FFT_ConvBlock(fea[4],fea[4])
        elif 'SE' in encModule:
            self.encModule1 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[0])
            self.encModule2 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[1])
            self.encModule3 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[2])
            self.encModule4 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[3])
            self.encModule5 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[4])
        elif 'CBAM' in encModule:
            self.encModule1 = CBAM(gate_channels=fea[0], reduction_ratio=16, pool_types=['avg', 'max'])
            self.encModule2 = CBAM(gate_channels=fea[1], reduction_ratio=16, pool_types=['avg', 'max'])
            self.encModule3 = CBAM(gate_channels=fea[2], reduction_ratio=16, pool_types=['avg', 'max'])
            self.encModule4 = CBAM(gate_channels=fea[3], reduction_ratio=16, pool_types=['avg', 'max'])
            self.encModule5 = CBAM(gate_channels=fea[4], reduction_ratio=16, pool_types=['avg', 'max'])
        elif 'MHA' in encModule:
            self.encModule1 = nn.MultiheadAttention(featureLength//2, 8, batch_first=True, dropout=0.01)
            self.encModule2 = nn.MultiheadAttention(featureLength//4, 8, batch_first=True, dropout=0.01)
            self.encModule3 = nn.MultiheadAttention(featureLength//8, 8, batch_first=True, dropout=0.01)
            self.encModule4 = nn.MultiheadAttention(featureLength//16, 8, batch_first=True, dropout=0.01)
            self.encModule5 = nn.MultiheadAttention(featureLength//32, 8, batch_first=True, dropout=0.01)
            
        # multiTaskCLS
        self.mtl = mtl   
        self.mtl_cls = nn.Identity()     
        self.mtl_rec = nn.Identity()     
        print(f"U-NET mtl is {mtl}")

        if mtl == "CLS" or "ALL" in mtl:
            self.mtl_cls = nn.Sequential(monai.networks.blocks.ResidualSELayer(spatial_dims, fea[4]),nn.AdaptiveAvgPool1d(1),nn.Conv1d(fea[4],1,1))

        if mtl == "REC" or "ALL" in mtl:
            mtl4 = Convolution(spatial_dims=spatial_dims, in_channels=fea[4], out_channels=fea[3], adn_ordering="ADN", act=("prelu", {"init": 0.2}), dropout=0.1, norm=norm)
            mtl3 = Convolution(spatial_dims=spatial_dims, in_channels=fea[3], out_channels=fea[2], adn_ordering="ADN", act=("prelu", {"init": 0.2}), dropout=0.1, norm=norm)
            mtl2 = Convolution(spatial_dims=spatial_dims, in_channels=fea[2], out_channels=fea[1], adn_ordering="ADN", act=("prelu", {"init": 0.2}), dropout=0.1, norm=norm)
            mtl1 = Convolution(spatial_dims=spatial_dims, in_channels=fea[1], out_channels=fea[0], adn_ordering="ADN", act=("prelu", {"init": 0.2}), dropout=0.1, norm=norm)
            mtl0 = Convolution(spatial_dims=spatial_dims, in_channels=fea[0], out_channels=in_channels, adn_ordering="ADN", act=("prelu", {"init": 0.2}), dropout=0.1, norm=norm)
            self.mtl_rec = nn.Sequential(mtl4,nn.Upsample(scale_factor=2),mtl3,nn.Upsample(scale_factor=2),mtl2,nn.Upsample(scale_factor=2),mtl1,nn.Upsample(scale_factor=2),mtl0,nn.Upsample(scale_factor=2))
        
        # U-Net Decoder
        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample, interp_mode='linear')
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample, interp_mode='linear')
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample, interp_mode='linear')
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[0], act, norm, bias, dropout, upsample, interp_mode='linear')
        self.upcat_0 = UpCat(spatial_dims, fea[0], fea[0], fea[0], act, norm, bias, dropout, upsample, interp_mode='linear', halves=False)

        self.decModule =  decModule
        print('U-NET decModule is', decModule)
        self.decModule1 = nn.Identity()
        self.decModule2 = nn.Identity()
        self.decModule3 = nn.Identity()
        self.decModule4 = nn.Identity()

        if 'SE' in self.decModule:
            self.decModule1 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[0]) # (128x256 and 512x256)
            self.decModule2 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[1])
            self.decModule3 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[2])
            self.decModule4 = monai.networks.blocks.ResidualSELayer(spatial_dims,fea[3])   

        elif 'NN' in self.decModule:
            self.decModule1 = NLBlockND(in_channels=fea[0], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.decModule2 = NLBlockND(in_channels=fea[1], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.decModule3 = NLBlockND(in_channels=fea[2], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.decModule4 = NLBlockND(in_channels=fea[3], mode='embedded', dimension=spatial_dims, norm_layer=norm)               

        elif 'FFC' in self.decModule:
            self.decModule1 = FFC_BN_ACT(fea[0],fea[0])
            self.decModule2 = FFC_BN_ACT(fea[1],fea[1])
            self.decModule3 = FFC_BN_ACT(fea[2],fea[2])
            self.decModule4 = FFC_BN_ACT(fea[3],fea[3])

        elif 'DEEPRFT' in self.decModule:
            self.decModule1 = FFT_ConvBlock(fea[0],fea[0])
            self.decModule2 = FFT_ConvBlock(fea[1],fea[1])
            self.decModule3 = FFT_ConvBlock(fea[2],fea[2])
            self.decModule4 = FFT_ConvBlock(fea[3],fea[3])
            
        elif 'ACM' in self.decModule:
            group = 4
            self.decModule1 = ACM(num_heads=fea[0]//group, num_features=fea[0], orthogonal_loss=False)
            self.decModule2 = ACM(num_heads=fea[1]//group, num_features=fea[1], orthogonal_loss=False)
            self.decModule3 = ACM(num_heads=fea[2]//group, num_features=fea[2], orthogonal_loss=False)
            self.decModule4 = ACM(num_heads=fea[3]//group, num_features=fea[3], orthogonal_loss=False)

        elif 'MHA' in self.decModule:
            # featureLength = 1280
            self.decModule1 = nn.MultiheadAttention(featureLength//2, 8, batch_first=True, dropout=0.01) 
            self.decModule2 = nn.MultiheadAttention(featureLength//4, 8, batch_first=True, dropout=0.01)
            self.decModule3 = nn.MultiheadAttention(featureLength//8, 8, batch_first=True, dropout=0.01)
            self.decModule4 = nn.MultiheadAttention(featureLength//16, 8, batch_first=True, dropout=0.01)

        self.supervision = supervision
        if supervision == 'NONE':
            supervision_c = fea[0]
        elif supervision =='TYPE1':
            supervision_c = fea[0]+fea[0]+fea[1]+fea[2]+fea[3]+fea[4]
        elif supervision =='TYPE2':
            self.sv0= Conv["conv", spatial_dims](fea[0], out_channels*8, kernel_size=3, padding=1)
            self.sv1= Conv["conv", spatial_dims](fea[0], out_channels*8, kernel_size=3, padding=1)
            self.sv2= Conv["conv", spatial_dims](fea[1], out_channels*8, kernel_size=3, padding=1)
            self.sv3= Conv["conv", spatial_dims](fea[2], out_channels*8, kernel_size=3, padding=1)
            self.sv4= Conv["conv", spatial_dims](fea[3], out_channels*8, kernel_size=3, padding=1)
            self.sv5= Conv["conv", spatial_dims](fea[4], out_channels*8, kernel_size=3, padding=1)
            supervision_c =  out_channels*8*6
            
        self.segheadModule = segheadModule
        self.segheadModule0 = nn.Identity()
        print(f"U-NET segheadModule is {segheadModule}")

        if 'ACM' in segheadModule:
            group = 4
            self.ACMLambda = 0
            if self.ACMLambda==0:
                self.segheadModule0 = ACM(num_heads=supervision_c//group, num_features=supervision_c, orthogonal_loss=False)
            else:
                self.segheadModule0 = ACM(num_heads=supervision_c//group, num_features=supervision_c, orthogonal_loss=True)

        elif 'NLNN' in segheadModule:
            norms = ['instance','batch']
            if not norm in norms:
                norm = None
            self.segheadModule0 = NLBlockND(in_channels=supervision_c, mode='embedded', dimension=spatial_dims, norm_layer=norm)

        elif 'FFC' in segheadModule:
            self.segheadModule0 = FFC_BN_ACT(supervision_c,supervision_c)       

        elif 'DEEPRFT' in segheadModule:
            self.segheadModule0 = FFT_ConvBlock(supervision_c,supervision_c)

        elif 'SE' in segheadModule:
            self.segheadModule0 = monai.networks.blocks.ResidualSELayer(spatial_dims, supervision_c)
            
        elif 'CBAM' in segheadModule:
            self.segheadModule0 = CBAM(gate_channels=supervision_c, reduction_ratio=16, pool_types=['avg', 'max'])

        elif 'MHA' in segheadModule:
            self.segheadModule0 = nn.MultiheadAttention(featureLength//1, 2, batch_first=True, dropout=0.01)

        # self.final_conv = nn.Sequential(self.segheadModule, Conv["conv", spatial_dims](supervision_c, out_channels, kernel_size=1),)
        self.final_conv = Conv["conv", spatial_dims](supervision_c, out_channels, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # set_seed()
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
        dp1 = dp2 = dp3 = dp4 = dp5 = 0.
        
        if self.encModule=="NONE":
            pass
        elif "MHA" in self.encModule:
            x1,_ = self.encModule1(x1,x1,x1)
            x2,_ = self.encModule2(x2,x2,x2)
            x3,_ = self.encModule3(x3,x3,x3)
            x4,_ = self.encModule4(x4,x4,x4)
            x5,_ = self.encModule5(x5,x5,x5)
        else:
            x1 = self.encModule1(x1)
            x2 = self.encModule2(x2)
            x3 = self.encModule3(x3)
            x4 = self.encModule4(x4)
            x5 = self.encModule5(x5)
        
#             x1 = x1 + self.encModule1(x1)
#             x2 = x2 + self.encModule2(x2)
#             x3 = x3 + self.encModule3(x3)
#             x4 = x4 + self.encModule4(x4)
#             x5 = x5 + self.encModule5(x5)
            
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
                                        
        out_cls = None    
        out_rec = None        
        if self.mtl_cls:
            out_cls = self.mtl_cls(x5)
        if self.mtl_rec:
            out_rec = self.mtl_rec(x5)
  
        u4 = self.upcat_4(x5, x4)
        if "MHA" not in self.decModule:
            u4 = self.decModule4(u4)
        else:
            u4,_ = self.decModule4(u4,u4,u4)
        u3 = self.upcat_3(u4, x3)
        if "MHA" not in self.decModule:
            u3 = self.decModule3(u3)
        else:
            u3,_ = self.decModule3(u3,u3,u3)
        u2 = self.upcat_2(u3, x2)
        if "MHA" not in self.decModule:
            u2 = self.decModule2(u2)
        else:
            u2,_ = self.decModule2(u2,u2,u2)
        u1 = self.upcat_1(u2, x1)
        if "MHA" not in self.decModule:
            u1 = self.decModule1(u1)
        else:
            u1,_ = self.decModule1(u1,u1,u1)
        u0 = self.upcat_0(u1, x0)
        # print(u0.shape, u1.shape, u2.shape, u3.shape, u4.shape)
        
        if self.supervision=='old' or self.supervision =='TYPE1':            
            s5 = _upsample_like(x5,u0)
            s4 = _upsample_like(x4,u0)
            s3 = _upsample_like(x3,u0)
            s2 = _upsample_like(x2,u0)
            s1 = _upsample_like(x1,u0)            
            u0 = torch.cat((u0,s1,s2,s3,s4,s5),dim=1)            
            # print(u0.shape, s1.shape, s2.shape, s3.shape, s4.shape, s5.shape)
            if 'MHA' in self.segheadModule:
                u0,_ = self.segheadModule0(u0,u0,u0)
            else:
                u0 = self.segheadModule0(u0)        
            logits = self.final_conv(u0)
            # print(logits.shape)
                                        
            if "ALL" in self.mtl:
                return [torch.sigmoid(logits), torch.sigmoid(out_cls), torch.tanh(out_rec)]
            elif self.mtl == "CLS":
                return [torch.sigmoid(logits), torch.sigmoid(out_cls)]
            elif self.mtl == "REC":
                return [torch.sigmoid(logits), torch.tanh(out_rec)]
            else:
                return torch.sigmoid(logits)

        elif self.supervision=='new' or self.supervision =='TYPE2':
            s5 = self.sv5(_upsample_like(x5,u0))
            s4 = self.sv4(_upsample_like(x4,u0))
            s3 = self.sv3(_upsample_like(x3,u0))
            s2 = self.sv2(_upsample_like(x2,u0))
            s1 = self.sv1(_upsample_like(x1,u0))
            u0 = self.sv0(u0)
            u0 = torch.cat((u0,s1,s2,s3,s4,s5),dim=1)
            # print(u0.shape, s1.shape, s2.shape, s3.shape, s4.shape, s5.shape)
            if 'MHA' in self.segheadModule:
                u0,_ = self.segheadModule0(u0,u0,u0)
            else:
                u0 = self.segheadModule0(u0)      
            logits = self.final_conv(u0)
            # print(logits.shape)

            if "ALL" in self.mtl:
                return [torch.sigmoid(logits), torch.sigmoid(out_cls), torch.tanh(out_rec)]
            elif self.mtl == "CLS":
                return [torch.sigmoid(logits), torch.sigmoid(out_cls)]
            elif self.mtl == "REC":
                return [torch.sigmoid(logits), torch.tanh(out_rec)]
            else:
                return torch.sigmoid(logits)
        else:
            if 'MHA' in self.segheadModule:
                u0,_ = self.segheadModule0(u0,u0,u0)
            else:
                u0 = self.segheadModule0(u0)     
            logits = self.final_conv(u0)
            # print(logits.shape)

            if "ALL" in self.mtl:
                return [torch.sigmoid(logits), torch.sigmoid(out_cls), torch.tanh(out_rec)]
            elif self.mtl == "CLS":
                return [torch.sigmoid(logits), torch.sigmoid(out_cls)]
            elif self.mtl == "REC":
                return [torch.sigmoid(logits), torch.tanh(out_rec)]
            else:
                return torch.sigmoid(logits)

    
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
                # x_0 = torch.nn.functional.pad(x_0, sp, "replicate") # original
                x_0 = torch.nn.functional.pad(x_0, sp, "constant") # for deterministic 
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x