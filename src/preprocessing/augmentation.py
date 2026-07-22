"""Data augmentation for training: random flips, rotations, and intensity jitter."""

import random

import numpy as np


def random_flip(image, p=0.5):
    if random.random() < p:
        image = np.fliplr(image)
    if random.random() < p:
        image = np.flipud(image)
    return image


def random_rotation(image, max_angle=15):
    from scipy.ndimage import rotate

    angle = random.uniform(-max_angle, max_angle)
    return rotate(image, angle, reshape=False, mode="nearest")


def random_intensity_jitter(image, scale=0.05):
    """Slightly perturb brightness/contrast; expects image already in [0, 1]."""
    factor = 1 + random.uniform(-scale, scale)
    offset = random.uniform(-scale, scale)
    return np.clip(image * factor + offset, 0, 1)


def augment(image):
    """Apply the full augmentation stack used during training."""
    image = random_flip(image)
    image = random_rotation(image)
    image = random_intensity_jitter(image)
    return image