import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # fft = self.norm1(fft)
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