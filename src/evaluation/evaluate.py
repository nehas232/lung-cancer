"""Run trained model on the held-out test set and write a performance report."""

import numpy as np
import torch

from src.config import CNN_WEIGHTS, DEVICE, OUTPUTS_DIR
from src.evaluation.metrics import compute_metrics
from src.models.cnn_model import load_model


def evaluate_on_test_set(test_loader, weights_path=CNN_WEIGHTS, device=None):
    device = device or (DEVICE if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, device=device)

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    write_report(metrics)
    return metrics


def write_report(metrics, path=None):
    path = path or (OUTPUTS_DIR / "evaluation_report.txt")
    with open(path, "w") as f:
        f.write("Lung cancer detection - evaluation report\n")
        f.write("=" * 45 + "\n")
        for key, value in metrics.items():
            f.write(f"{key:20s}: {value}\n")
    print(f"Report written to {path}")


if __name__ == "__main__":
    print("Provide a test DataLoader to evaluate_on_test_set(...) to run evaluation.")