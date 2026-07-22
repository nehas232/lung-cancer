"""Extract shape, intensity, and texture features from a candidate nodule patch.

These handcrafted features can feed a classical ML classifier (e.g. random
forest) or be logged alongside CNN predictions for interpretability.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_patch(image, y, x, radius, margin=5):
    """Crop a square patch around a candidate center."""
    r = int(radius) + margin
    y0, y1 = max(0, y - r), y + r
    x0, x1 = max(0, x - r), x + r
    return image[y0:y1, x0:x1]


def shape_features(radius):
    return {
        "diameter_px": radius * 2,
        "area_px": np.pi * radius ** 2,
    }


def intensity_features(patch):
    return {
        "mean_hu": float(np.mean(patch)),
        "std_hu": float(np.std(patch)),
        "max_hu": float(np.max(patch)),
        "min_hu": float(np.min(patch)),
    }


def texture_features(patch, distances=(1,), angles=(0,)):
    """GLCM-based texture descriptors: contrast, homogeneity, energy, correlation."""
    normalized = np.clip(patch, patch.min(), patch.max())
    scaled = ((normalized - normalized.min()) / (normalized.ptp() + 1e-8) * 255).astype(np.uint8)

    glcm = graycomatrix(scaled, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    return {
        "contrast": float(graycoprops(glcm, "contrast").mean()),
        "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "energy": float(graycoprops(glcm, "energy").mean()),
        "correlation": float(graycoprops(glcm, "correlation").mean()),
    }


def extract_features(image, y, x, radius):
    """Combine shape, intensity, and texture features for one candidate."""
    patch = extract_patch(image, y, x, radius)
    features = {}
    features.update(shape_features(radius))
    features.update(intensity_features(patch))
    features.update(texture_features(patch))
    return features