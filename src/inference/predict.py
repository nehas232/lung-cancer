"""Run the full inference pipeline on a single new DICOM scan."""

import numpy as np
import torch

from src.config import CNN_WEIGHTS, DEVICE, IMAGE_SIZE
from src.detection.nodule_detector import detect_candidate_nodules, filter_by_size
from src.models.cnn_model import load_model
from src.preprocessing.load_dicom import load_dicom_slice
from src.preprocessing.lung_segmentation import segment_lungs, apply_lung_mask
from src.preprocessing.normalize import normalize_to_unit_range, resize_image


def predict_scan(filepath, weights_path=CNN_WEIGHTS, device=None):
    """Full pipeline: load -> preprocess -> detect candidates -> classify each."""
    device = device or (DEVICE if torch.cuda.is_available() else "cpu")

    hu_image, _ = load_dicom_slice(filepath)
    mask = segment_lungs(hu_image)
    lung_only = apply_lung_mask(hu_image, mask)
    normalized = normalize_to_unit_range(lung_only)

    candidates = filter_by_size(detect_candidate_nodules(normalized))

    model = load_model(weights_path, device=device)

    results = []
    for y, x, radius in candidates:
        patch = crop_and_resize(normalized, y, x, radius)
        prob_malignant = classify_patch(model, patch, device)
        results.append({
            "y": y, "x": x, "radius": radius,
            "malignancy_probability": prob_malignant,
        })

    return results


def crop_and_resize(image, y, x, radius, margin=5):
    r = int(radius) + margin
    y0, y1 = max(0, y - r), y + r
    x0, x1 = max(0, x - r), x + r
    patch = image[y0:y1, x0:x1]
    return resize_image(patch, IMAGE_SIZE)


def classify_patch(model, patch, device):
    tensor = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()
    return prob


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.predict <path_to_dicom_file>")
    else:
        findings = predict_scan(sys.argv[1])
        for f in findings:
            print(f)