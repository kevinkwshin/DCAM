
import torch
import torch.nn as nn
import torch.nn.functional as F

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:],mode='linear')

    return src



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

    def __init__(self,in_ch=1,out_ch=1,nnblock=False, FFC=False, acm=False, ASPP=False, temperature=1, dropout=0.1, norm='instance'):
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
        self.nnblock = nnblock
        if nnblock:
            spatial_dims = 1
            self.nnblock1 = NLBlockND(in_channels=fea[0], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.nnblock2 = NLBlockND(in_channels=fea[1], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.nnblock3 = NLBlockND(in_channels=fea[2], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.nnblock4 = NLBlockND(in_channels=fea[3], mode='embedded', dimension=spatial_dims, norm_layer=norm)
            self.nnblock5 = NLBlockND(in_channels=fea[4], mode='embedded', dimension=spatial_dims, norm_layer=norm)     
            self.nnblock6 = NLBlockND(in_channels=fea[5], mode='embedded', dimension=spatial_dims, norm_layer=norm)                      

        self.FFC = FFC
        if FFC=='FFC':
            self.FFCblock1 = FFC_BN_ACT(fea[0],fea[0])
            self.FFCblock2 = FFC_BN_ACT(fea[1],fea[1])
            self.FFCblock3 = FFC_BN_ACT(fea[2],fea[2])
            self.FFCblock4 = FFC_BN_ACT(fea[3],fea[3])
            self.FFCblock5 = FFC_BN_ACT(fea[4],fea[4])
            self.FFCblock6 = FFC_BN_ACT(fea[5],fea[5])            
        elif FFC=='DeepRFT':
            self.FFCblock1 = FFT_ConvBlock(fea[0],fea[0])
            self.FFCblock2 = FFT_ConvBlock(fea[1],fea[1])
            self.FFCblock3 = FFT_ConvBlock(fea[2],fea[2])
            self.FFCblock4 = FFT_ConvBlock(fea[3],fea[3])
            self.FFCblock5 = FFT_ConvBlock(fea[4],fea[4])
            self.FFCblock6 = FFT_ConvBlock(fea[5],fea[5])
            
        self.acm = acm
        if acm:
            self.acm1 = ACM(num_heads=fea[0]//2, num_features=fea[0], orthogonal_loss=False)
            self.acm2 = ACM(num_heads=fea[1]//2, num_features=fea[1], orthogonal_loss=False)
            self.acm3 = ACM(num_heads=fea[2]//2, num_features=fea[2], orthogonal_loss=False)
            self.acm4 = ACM(num_heads=fea[3]//2, num_features=fea[3], orthogonal_loss=False)
            self.acm5 = ACM(num_heads=fea[4]//2, num_features=fea[4], orthogonal_loss=False)
            self.acm6 = ACM(num_heads=fea[5]//2, num_features=fea[5], orthogonal_loss=False)
            
        self.ASPP = ASPP
        spatial_dims = 1
        if ASPP=='last':
            self.ASPPblock6 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[5], conv_out_channels=fea[5]//4,
                                                               norm_type=norm, acti_type='LEAKYRELU', bias=False)  
        elif ASPP=='all':
            self.ASPPblock1 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[0], conv_out_channels=fea[0]//4,
                                                               norm_type=norm, acti_type='LEAKYRELU', bias=False)  
            self.ASPPblock2 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[1], conv_out_channels=fea[1]//4,
                                                               norm_type=norm, acti_type='LEAKYRELU', bias=False)  
            self.ASPPblock3 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[2], conv_out_channels=fea[2]//4,
                                                               norm_type=norm, acti_type='LEAKYRELU', bias=False)  
            self.ASPPblock4 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[3], conv_out_channels=fea[3]//4,
                                                               norm_type=norm, acti_type='LEAKYRELU', bias=False) 
            self.ASPPblock5 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[4], conv_out_channels=fea[4]//4,
                                                               norm_type=norm, acti_type='LEAKYRELU', bias=False)  
            self.ASPPblock6 = monai.networks.blocks.SimpleASPP(spatial_dims=spatial_dims, in_channels=fea[5], conv_out_channels=fea[5]//4,
                                                               norm_type=norm, acti_type='LEAKYRELU', bias=False)  
        self.temperature = temperature
            
    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx1 = hx1 + self.ASPPblock1(hx1) if self.ASPP=='all' else hx1
        hx1 = hx1 + self.nnblock1(hx1) if self.nnblock else hx1
        hx1 = hx1 + self.FFCblock1(hx1) if self.FFC else hx1
        hx1 = hx1 + self.acm1(hx1) if self.acm else hx1
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx2 = hx2 + self.ASPPblock2(hx2) if self.ASPP=='all' else hx2
        hx2 = hx2 + self.nnblock2(hx2) if self.nnblock else hx2
        hx2 = hx2 + self.FFCblock2(hx2) if self.FFC else hx2
        hx2 = hx2 + self.acm2(hx2) if self.acm else hx2
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx3 = hx3 + self.ASPPblock3(hx3) if self.ASPP=='all' else hx3
        hx3 = hx3 + self.nnblock3(hx3) if self.nnblock else hx3
        hx3 = hx3 + self.FFCblock3(hx3) if self.FFC else hx3
        hx3 = hx3 + self.acm3(hx3) if self.acm else hx3
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx4 = hx4 + self.ASPPblock4(hx4) if self.ASPP=='all' else hx4
        hx4 = hx4 + self.nnblock4(hx4) if self.nnblock else hx4
        hx4 = hx4 + self.FFCblock4(hx4) if self.FFC else hx4
        hx4 = hx4 + self.acm4(hx4) if self.acm else hx4
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx5 = hx5 + self.ASPPblock5(hx5) if self.ASPP=='all' else hx5
        hx5 = hx5 +self.nnblock5(hx5) if self.nnblock else hx5
        hx5 = hx5 +self.FFCblock5(hx5) if self.FFC else hx5
        hx5 = hx5 +self.acm5(hx5) if self.acm else hx5
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6 = hx6 + self.ASPPblock6(hx6) if self.ASPP else hx6
        hx6 = hx6 + self.nnblock6(hx6) if self.nnblock else hx6
        hx6 = hx6 + self.FFCblock6(hx6) if self.FFC else hx6
        hx6 = hx6 + self.acm6(hx6) if self.acm else hx6
        hx6up = _upsample_like(hx6,hx5)
        
        #-------------------- decoder --------------------
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

def muti_bce_loss_fusion(yhat, y):
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

lossfn = muti_bce_loss_fusion