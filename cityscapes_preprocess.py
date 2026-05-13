import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False


def normalize_rgb_image(img_array: np.ndarray, imagenet_normalize: bool = False) -> np.ndarray:
    """Normalize an RGB image to match the training/inference pipeline."""
    normalized = img_array.astype(np.float32)
    normalized = normalized / 255.0

    if imagenet_normalize:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

    return normalized


def resize_rgb(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize RGB image with bilinear interpolation."""
    if CV2_AVAILABLE:
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.array(Image.fromarray(img).resize((width, height), Image.BILINEAR))


def resize_mask_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize label mask with nearest-neighbor interpolation."""
    if CV2_AVAILABLE:
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.array(Image.fromarray(mask).resize((width, height), Image.NEAREST))


def save_rgb_png(path: str, rgb_img: np.ndarray) -> None:
    """Save an RGB image using OpenCV if available, otherwise Pillow."""
    if CV2_AVAILABLE:
        cv2.imwrite(path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    else:
        Image.fromarray(rgb_img).save(path)