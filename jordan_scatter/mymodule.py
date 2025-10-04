import torch
import torch.nn as nn
from torch.fft import fft2, ifft2

class mymodule(nn.Module):
    def __init__(self, max_scale:int, nb_orients:int, # number of scales and orientations
                 image_size:int, # square image, width=height
                 filters,
                 mixed_precision: bool = False):
        super().__init__()
        self.max_scale = max_scale
        self.nb_orients = nb_orients
        self.image_size = image_size
        self.mixed_precision = mixed_precision
        # keep filters as buffers so they follow the module to the active device
        self.register_buffer("lp", filters["lp"].clone())
        self.register_buffer("hp", filters["hp"].clone())

        # verify shape
        if self.hp.shape != (max_scale, nb_orients, image_size, image_size):
            raise ValueError(f"expected high pass shape: [J,K,N,N], but get {self.hp.shape}")
        if self.lp.shape != (image_size, image_size):
            raise ValueError(f"expected low pass shape: [1,1,N,N], but get {self.lp.shape}")
    
    def forward(self, x):
        # verify the shape
        if x.dim() != 6:
            raise ValueError("Input should have 6 dimensions")
        orig_real_dtype = x.dtype
        use_amp = (
            self.mixed_precision
            and x.device.type in {"cuda", "mps"}
            and orig_real_dtype in {torch.float32, torch.float16}
        )
        compute_real_dtype = orig_real_dtype
        if use_amp and orig_real_dtype == torch.float32:
            compute_real_dtype = torch.float16
        if compute_real_dtype != orig_real_dtype:
            x = x.to(compute_real_dtype)
        # compute high pass and low pass
        x_hat = fft2(x)
        lp = self.lp.to(x_hat.dtype)
        hp = self.hp.to(x_hat.dtype)
        x_lp_hat = x_hat*lp
        x_hp_hat = x_hat*hp
        del x_hat

        # compute jordan
        x_hp = ifft2(x_hp_hat)
        del x_hp_hat

        x_hp_real = x_hp.real
        x_hp_imag = x_hp.imag
        del x_hp

        batch, channel, scales, orients, height, width = x_hp_real.shape
        y = torch.empty((batch, channel * 4, scales, orients, height, width), device=x_hp_real.device, dtype=x_hp_real.dtype)

        torch.clamp(x_hp_real, min=0, out=y[:, :channel])
        torch.clamp(x_hp_real, max=0, out=y[:, channel:2*channel]).neg_()
        torch.clamp(x_hp_imag, min=0, out=y[:, 2*channel:3*channel])
        torch.clamp(x_hp_imag, max=0, out=y[:, 3*channel:]).neg_()

        del x_hp_real, x_hp_imag

        y = y.reshape(batch, -1, 1, 1, height, width)
        if compute_real_dtype != orig_real_dtype:
            y = y.to(orig_real_dtype)
            target_complex = torch.complex64 if orig_real_dtype == torch.float32 else torch.complex32
            x_lp_hat = x_lp_hat.to(target_complex)
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
