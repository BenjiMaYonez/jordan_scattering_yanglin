import torch
import torch.nn as nn
from torch.fft import fft2, ifft2

class mymodule(nn.Module):
    def __init__(self, max_scale:int, nb_orients:int, # number of scales and orientations
                 image_size:int, # square image, width=height
                 filters):
        super().__init__()
        self.max_scale = max_scale
        self.nb_orients = nb_orients
        self.image_size = image_size
        self.lp = filters["lp"] # low pass filter
        self.hp = filters["hp"] # high pass filter

        # verify shape
        if self.hp.shape != (max_scale, nb_orients, image_size, image_size):
            raise ValueError(f"expected high pass shape: [J,K,N,N], but get {self.hp.shape}")
        if self.lp.shape != (image_size, image_size):
            raise ValueError(f"expected low pass shape: [1,1,N,N], but get {self.lp.shape}")
    
    def forward(self, x):
        # verify the shape
        if x.dim() != 6:
            raise ValueError("Input should have 6 dimensions")
        # compute high pass and low pass
        x_hat = fft2(x)
        x_lp_hat = x_hat*self.lp
        x_hp_hat = x_hat*self.hp

        # compute jordan
        x_hp = ifft2(x_hp_hat)
        x_hp_real_pos = torch.relu(x_hp.real)
        x_hp_real_neg = torch.relu(-x_hp.real)
        x_hp_imag_pos = torch.relu(x_hp.imag)
        x_hp_imag_neg = torch.relu(-x_hp.imag)
        x_jordan = torch.cat([x_hp_real_pos, x_hp_real_neg, 
                              x_hp_imag_pos, x_hp_imag_neg], dim=1)
        channel = x_jordan.size(1)*x_jordan.size(2)*x_jordan.size(3)
        y = x_jordan.reshape(x_jordan.size(0), channel, 1, 1, self.image_size, self.image_size)
        return x_lp_hat, y

    def inverse(self, x_lp_hat, y):
        # verify the shape
        if (x_lp_hat.dim() != 6 or y.dim() != 6):
            raise ValueError(f"Expected inputs to have 6 dims, but get {x_lp_hat.dim()} and {y.dim()}")

        # inverse jordan
        x_jordan = y.reshape(1, -1, self.max_scale, self.nb_orients, self.image_size, self.image_size)
        if x_jordan.size(1)%4 != 0:
            raise ValueError(f"expected channel 1 to be multiple of 4, but get {x_jordan.size(1)}")
        channel = x_jordan.size(1)//4
        x_hp_real_pos, x_hp_real_neg, x_hp_imag_pos, x_hp_imag_neg = torch.split(x_jordan, channel, dim=1)
        x_hp = torch.complex(x_hp_real_pos-x_hp_real_neg, x_hp_imag_pos-x_hp_imag_neg)
        x_hp_hat = fft2(x_hp)

        # inverse wavelet
        recx_hp_hat = (x_hp_hat*self.hp).sum(dim=(-4,-3), keepdim=True)
        recx_lp_hat = x_lp_hat*self.lp
        x_hat = recx_hp_hat + recx_lp_hat
        x = ifft2(x_hat)

        return x.real