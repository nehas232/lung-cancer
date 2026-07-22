"""Download the lung cancer chest X-ray dataset from Kaggle via kagglehub."""

import os
import kagglehub

DATASET_SLUG = "quynhlecl/lung-cancer-x-ray"


def download_dataset():
    """Download the dataset (or return cached path if already downloaded)."""
    path = kagglehub.dataset_download(DATASET_SLUG)
    print("Path to dataset files:", path)
    return path


def print_tree(path, max_depth=3):
    """Print folder structure up to max_depth, showing a couple sample files per folder."""
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < max_depth:
            for f in files[:2]:
                print(f"{indent}  {f}")


if __name__ == "__main__":
    dataset_path = download_dataset()
    print("\nFolder structure:\n")
    print_tree(dataset_path)