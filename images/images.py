import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from jordan_scatter.helpers import LoggerManager

def load_images(image_folder: str, color: bool = False) -> torch.Tensor:
    """
    Load all .png and .jpg images from a folder, verify they are square and same size,
    convert to tensor [B, C, N, N].
    
    Args:
        image_folder: folder containing images
        color: True -> 3 channels (RGB), False -> 1 channel (grayscale)
    """
    image_folder = Path(image_folder)
    if not image_folder.exists():
        raise ValueError(f"Folder {image_folder} does not exist.")
    
    image_paths = list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpg"))
    if len(image_paths) == 0:
        raise ValueError("No .png or .jpg images found in folder.")
    
    images = []
    size = None
    
    for path in image_paths:
        img = Image.open(path).convert("RGB" if color else "L")  # RGB or grayscale
        if img.width != img.height:
            raise ValueError(f"Image {path} is not square: {img.width}x{img.height}")
        if size is None:
            size = img.width
        elif img.width != size:
            raise ValueError(f"Image {path} size {img.width} does not match other images {size}.")
        
        img_tensor = transforms.ToTensor()(img)  # [C, H, W], values in [0,1]
        images.append(img_tensor)
    logger = LoggerManager.get_logger()
    logger.info(f"Load {len(images)} images from {image_folder}")
    # stack into [B, C, N, N]
    return torch.stack(images, dim=0)


def save_images(save_folder: str, tensor: torch.Tensor):
    """
    Save tensor [B, C, N, N] as individual images to save_folder.
    Tensor might be on GPU; will convert to CPU automatically.
    """
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    
    # move to cpu and clamp to [0,1] just in case
    tensor = tensor.detach().cpu().clamp(0, 1)
    logger = LoggerManager.get_logger()
    B = tensor.shape[0]
    logger.info(f"Saving {B} images to {save_folder}")
    for i in range(B):
        img_tensor = tensor[i]
        img = transforms.ToPILImage()(img_tensor)
        img.save(save_folder / f"img_{i:03d}.png")
