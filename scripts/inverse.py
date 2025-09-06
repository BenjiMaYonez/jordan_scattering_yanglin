import argparse
from configs import load_config, save_config
from images import load_images, save_images
import os
from datetime import datetime
import torch
from jordan_scatter.helpers import LoggerManager

from jordan_scatter import jordan_scatter
def parse_args():
    parser = argparse.ArgumentParser(description="Inverse the jordan scatter coefficients with config")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/full_inverse.yaml)"
    )
    return parser.parse_args()

def main():
    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join("experiments", f"run-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # init logger
    logger = LoggerManager.get_logger(log_dir=exp_dir)
    logger.info("Start log:")

    # Read config and backup config
    args = parse_args()
    config = load_config(args.config)
    save_config(exp_dir, config)

    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load images
    image_dir = os.path.join("images")
    images = load_images(image_dir)
    batch, channel, image_size, image_size2 = images.shape
    image_shape = channel, image_size,image_size2
    
    # init model
    max_scale = config["max_scale"]
    nb_orients = config["nb_orients"]
    depth = config["depth"]
    wavelet = config["wavelet"]
    model = jordan_scatter(max_scale, nb_orients, image_shape, depth, wavelet=wavelet)

    # Running
    logger.info("Run forward...")
    images = images.reshape(batch, channel, 1, 1, image_size, image_size)
    output, img = model(images)

    full_inverse = config["full_inverse"]
    if full_inverse:
        logger.info('Run lossless inversion...')
        rec_img = model.inverse(output, img)
        save_images(exp_dir, rec_img.real)
    else:
        logger.info('Run lossy inversion...')
        img = torch.zeros_like(img)
        rec_img = model.inverse(output, img)
        save_images(exp_dir, rec_img.real)

if __name__ == "__main__":
    main()