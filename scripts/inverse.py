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
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Metal (MPS)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    logger.info(f"Using device: {device} ({device_name})")

    # Load images (keep on CPU for graceful fallback)
    image_dir = os.path.join("images")
    images = load_images(image_dir)
    batch, channel, image_size, image_size2 = images.shape
    image_shape = channel, image_size,image_size2
    
    # init model
    max_scale = config["max_scale"]
    nb_orients = config["nb_orients"]
    depth = config["depth"]
    wavelet = config["wavelet"]
    normalize_wavelets = config.get("normalize_wavelets", True)
    optim_config = config.get("optimizations", {})
    use_mixed_precision = bool(optim_config.get("mixed_precision", False))
    use_disk_cache = bool(optim_config.get("disk_cache", False))
    disk_cache_dir = optim_config.get("disk_cache_dir")
    env_limit = os.environ.get("JORDAN_DEVICE_TENSOR_LIMIT_GB")
    if env_limit is not None:
        try:
            tensor_limit_bytes = int(float(env_limit) * (1024 ** 3))
        except ValueError:
            raise ValueError("JORDAN_DEVICE_TENSOR_LIMIT_GB must be numeric")
    elif device.type != "cpu":
        tensor_limit_bytes = 8 * (1024 ** 3)
    else:
        tensor_limit_bytes = None

    model = jordan_scatter(
        max_scale,
        nb_orients,
        image_shape,
        depth,
        wavelet=wavelet,
        normalize_wavelets=normalize_wavelets,
        device_tensor_limit_bytes=tensor_limit_bytes,
        use_mixed_precision=use_mixed_precision,
        use_disk_cache=use_disk_cache,
        disk_cache_dir=disk_cache_dir,
    )

    # Attempt execution on preferred device, fall back to CPU on memory pressure
    target_devices = [(device, device_name)]
    if device.type != "cpu":
        target_devices.append((torch.device("cpu"), "CPU"))

    last_error = None
    rec_img = None
    actual_device = None
    try:
        for target_device, target_name in target_devices:
            try:
                logger.info(f"Run forward on {target_name}...")
                model = model.to(target_device)
                images_device = images.to(target_device)
                images_device = images_device.reshape(batch, channel, 1, 1, image_size, image_size)

                with torch.no_grad():
                    output, img = model(images_device)

                    full_inverse = config["full_inverse"]
                    if full_inverse:
                        logger.info('Run lossless inversion...')
                        rec_img = model.inverse(output, img)
                    else:
                        logger.info('Run lossy inversion...')
                        img = torch.zeros_like(img)
                        rec_img = model.inverse(output, img)

                actual_device = target_name
                break
            except RuntimeError as err:
                err_msg = str(err).lower()
                oom_signature = (
                    "out of memory" in err_msg
                    or "command buffer exited" in err_msg
                    or "invalid buffer size" in err_msg
                )
                if target_device.type == "cpu" or not oom_signature:
                    raise
                last_error = err
                logger.warning(f"{target_name} ran out of memory, retrying on CPU...")
                model.cleanup_cache()
                if target_device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if target_device.type == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()
                model = model.to(torch.device("cpu"))
            finally:
                if 'images_device' in locals():
                    del images_device

        if rec_img is None:
            raise last_error if last_error is not None else RuntimeError("Forward pass did not produce an output.")

        if actual_device and actual_device != device_name:
            logger.info(f"Final computation ran on {actual_device}")

        save_images(exp_dir, rec_img.real)
    finally:
        if hasattr(model, "cleanup_cache"):
            model.cleanup_cache()

if __name__ == "__main__":
    main()
