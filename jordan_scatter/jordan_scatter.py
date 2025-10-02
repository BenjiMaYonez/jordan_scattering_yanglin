import torch
from torch import nn
from .wavelets import filter_bank
from .mymodule import mymodule
from .helpers import LoggerManager
class jordan_scatter(nn.Module):
    def __init__(self, max_scale:int, nb_orients:int, # number of scales and orientations
                       image_shape:tuple,        # image_channel, Height, Width
                       depth:int,                # depth of the network
                       wavelet:str = "morlet",   # Choose wavelet to use
                       normalize_wavelets:bool = True,
                       device_tensor_limit_bytes:int | None = None,
                ):
        super().__init__()
        self.max_scale = max_scale
        self.nb_orients = nb_orients
        self.image_channel, self.image_size, self.image_size2 = image_shape
        self.wavelet = wavelet
        self.depth = depth
        self.normalize_wavelets = normalize_wavelets
        self.device_tensor_limit_bytes = device_tensor_limit_bytes

        # check valid parameter
        if self.image_size != self.image_size2:
            raise ValueError(f"Images are not square size: {self.image_size}, {self.image_size2}")
        if depth <= 0:
            raise ValueError(f"{depth=} must be positive")
        
        # prepare filters of different scales and angles
        self.set_filters()

        # prepare module
        self.set_modules()

    def set_filters(self):
        filters = filter_bank(
            self.wavelet,
            self.max_scale,
            self.nb_orients,
            self.image_size,
            normalize=self.normalize_wavelets,
        )
        self.register_buffer("hp", filters["hp"])
        self.register_buffer("lp", filters["lp"])

    def set_modules(self):
        max_scale = self.max_scale
        nb_orients = self.nb_orients
        image_size = self.image_size
        filters = {"hp": self.hp, "lp": self.lp} # must first register in set_filters then do this
        self.module_list = nn.ModuleList()
        for layer in range(self.depth):
            module = mymodule(max_scale, nb_orients, image_size, filters)
            self.module_list.append(module)
        
    def forward(self, img):
        output = []
        logger = LoggerManager.get_logger()
        for layer_idx, module in enumerate(self.module_list):
            if (
                img.device.type != "cpu"
                and self.device_tensor_limit_bytes is not None
            ):
                next_channels = img.size(1) * self.max_scale * self.nb_orients * 4
                next_bytes = (
                    img.size(0)
                    * next_channels
                    * self.image_size
                    * self.image_size
                    * img.element_size()
                )
                if next_bytes > self.device_tensor_limit_bytes:
                    logger.warning(
                        "Layer %d tensor would require %.2f GiB on %s; continuing on CPU",
                        layer_idx,
                        next_bytes / (1024 ** 3),
                        img.device,
                    )
                    self.to("cpu")
                    img = img.cpu()
                    output = [tensor.cpu() for tensor in output]
            x_lp_hat, img = module(img)
            output.append(x_lp_hat)
        return output, img
    
    def inverse(self, output, img):
        for rev_ind, module in enumerate(reversed(self.module_list)):
            x_lp_hat = output[-rev_ind-1]
            img = module.inverse(x_lp_hat, img)
        logger = LoggerManager.get_logger()
        logger.info(f"Finish inversion, shape: {img.shape}. Return squeezed dim=(-4, -3).")
        return img.squeeze(-3).squeeze(-3)
            
