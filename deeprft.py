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