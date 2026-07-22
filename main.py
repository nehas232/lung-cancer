"""Entry point for the lung cancer chest X-ray classifier.

Usage:
    # Train on data/raw/lung-cancer-xray/chest_xray_lung/{train,val}
    python main.py --mode train

    # Evaluate on the test split
    python main.py --mode evaluate

    # Predict on a single image
    python main.py --mode predict --input path/to/image.jpg
"""

import argparse
import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.config import DEVICE, BATCH_SIZE, IMAGE_SIZE, CNN_WEIGHTS
from src.models.cnn_model import LungCancerCNN, load_model
from src.models.train_model import train
from src.evaluation.evaluate import evaluate_on_test_set
from src.inference.generate_report import generate_report

DATA_ROOT = os.path.join("data", "raw", "lung-cancer-xray", "chest_xray_lung")

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def get_loader(split, shuffle=False):
    """Load a train/val/test split as an ImageFolder DataLoader.

    Expects: DATA_ROOT/<split>/Cancer/*.jpg, DATA_ROOT/<split>/NORMAL/*.jpg
    """
    split_dir = os.path.join(DATA_ROOT, split)
    dataset = datasets.ImageFolder(split_dir, transform=TRANSFORM)
    print(f"Loaded {len(dataset)} images from {split_dir} | classes: {dataset.classes}")
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def run_train(device):
    train_loader = get_loader("train", shuffle=True)
    val_loader = get_loader("val", shuffle=False)

    model = LungCancerCNN().to(device)
    train(model, train_loader, val_loader, device)


def run_evaluate(device):
    test_loader = get_loader("test", shuffle=False)
    metrics = evaluate_on_test_set(test_loader, weights_path=CNN_WEIGHTS, device=device)
    print(metrics)


def run_predict(image_path, device):
    from PIL import Image

    model = load_model(CNN_WEIGHTS, device=device)
    image = Image.open(image_path)
    tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()

    patient_id = os.path.splitext(os.path.basename(image_path))[0]
    finding = [{"y": 0, "x": 0, "radius": 0, "malignancy_probability": prob}]
    generate_report(patient_id, finding)
    print(f"Predicted malignancy probability: {prob:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Lung cancer X-ray classifier")
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], required=True)
    parser.add_argument("--input", help="Path to an image file (required for --mode predict)")
    args = parser.parse_args()

    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.mode == "train":
        run_train(device)
    elif args.mode == "evaluate":
        run_evaluate(device)
    elif args.mode == "predict":
        if not args.input:
            parser.error("--input is required for --mode predict")
        run_predict(args.input, device)


if __name__ == "__main__":
    main()