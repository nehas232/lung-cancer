"""Normalize CT images: HU windowing and rescaling to [0, 1]."""

import numpy as np

from src.config import HU_MIN, HU_MAX


def clip_hu(image, hu_min=HU_MIN, hu_max=HU_MAX):
    """Clip Hounsfield units to a lung-relevant window."""
    return np.clip(image, hu_min, hu_max)


def normalize_to_unit_range(image, hu_min=HU_MIN, hu_max=HU_MAX):
    """Rescale a HU-clipped image to floating point [0, 1]."""
    image = clip_hu(image, hu_min, hu_max)
    return (image - hu_min) / (hu_max - hu_min)


def resize_image(image, target_size):
    """Resize a 2D image to (H, W) using bilinear interpolation."""
    import cv2

    return cv2.resize(image.astype(np.float32), (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)