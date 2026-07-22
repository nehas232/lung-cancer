"""Training loop for the CNN classifier."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import BATCH_SIZE, DEVICE, LEARNING_RATE, NUM_EPOCHS, CNN_WEIGHTS
from src.models.cnn_model import LungCancerCNN


class NoduleDataset(Dataset):
    """Loads preprocessed nodule patches and their benign/malignant labels."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import numpy as np

        image = np.load(self.image_paths[idx]).astype("float32")
        if self.transform:
            image = self.transform(image)
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_tensor, label


def train(model, train_loader, val_loader, device, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CNN_WEIGHTS)
            print(f"  Saved new best model to {CNN_WEIGHTS}")

    return model


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


if __name__ == "__main__":
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = LungCancerCNN().to(device)
    print("Model initialized. Provide train/val DataLoaders to start training.")