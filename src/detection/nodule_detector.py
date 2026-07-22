"""Candidate nodule detection within a segmented lung region.

Uses simple blob detection as a baseline candidate generator; in practice
this stage is often replaced or complemented by a learned detector
(e.g. a 3D CNN or Faster R-CNN operating on the segmented volume).
"""

from skimage.feature import blob_log


def detect_candidate_nodules(lung_image, min_sigma=2, max_sigma=10, threshold=0.05):
    """Find round, blob-like candidates using Laplacian-of-Gaussian detection.

    Returns a list of (y, x, radius) tuples for each candidate.
    """
    blobs = blob_log(lung_image, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    candidates = [(y, x, sigma * (2 ** 0.5)) for y, x, sigma in blobs]
    return candidates


def filter_by_size(candidates, min_radius=3, max_radius=30):
    """Discard candidates outside a plausible nodule size range (in pixels)."""
    return [c for c in candidates if min_radius <= c[2] <= max_radius]