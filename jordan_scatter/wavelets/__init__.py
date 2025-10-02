from .morlet import filter_bank as morlet
from .meyer import filter_bank as meyer
from ..helpers import LoggerManager


def filter_bank(
    wavelet_name: str,
    max_scale: int,
    nb_orients: int,
    image_size: int,
    normalize: bool = True,
):
    """
    Returns a dictionary, key "lp": low pass filter
                          key "hp": high pass filter
    Args:
        wavelet_name (str): "morlet", ...
        max_scale (int): maximum scales
        nb_orients (int): total orientations
        image_size (int): square image size, width=height
    """
    logger = LoggerManager.get_logger()
    logger.info(f"Select {wavelet_name} wavelet")

    wavelets_map = {
        "morlet": morlet,
        "meyer": meyer,
    }
    
    if wavelet_name not in wavelets_map:
        raise ValueError(f"Unknown wavelet: {wavelet_name}")
    
    wavelet_func = wavelets_map[wavelet_name]
    return wavelet_func(max_scale, nb_orients, image_size, normalize=normalize)



__all__ = ["filter_bank"]
