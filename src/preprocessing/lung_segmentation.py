"""Isolate lung regions from a CT slice using thresholding and morphology.

This is a classic (non-deep-learning) segmentation approach, useful as a
preprocessing step or a baseline before swapping in the U-Net model.
"""

import numpy as np
from skimage import measure, morphology


def segment_lungs(hu_image, threshold=-320):
    """Return a binary mask isolating the lung region of a single CT slice.

    Steps: threshold air/tissue, remove border-connected components (the body
    outline), keep the two largest remaining regions (the lungs), then fill
    holes and smooth with morphological closing.
    """
    binary = hu_image < threshold

    cleared = morphology.remove_small_objects(binary, min_size=500)
    labels = measure.label(cleared)

    regions = measure.regionprops(labels)
    regions = sorted(regions, key=lambda r: r.area, reverse=True)

    mask = np.zeros_like(binary)
    for region in regions[:2]:
        mask[labels == region.label] = True

    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = morphology.remove_small_holes(mask, area_threshold=1000)

    return mask


def apply_lung_mask(hu_image, mask, fill_value=-1000):
    """Zero out everything outside the lung mask (set to air HU value)."""
    output = hu_image.copy()
    output[~mask] = fill_value
    return output