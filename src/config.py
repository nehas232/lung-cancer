"""Central configuration: paths, image settings, and training hyperparameters."""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_SAVED_DIR = ROOT_DIR / "models_saved"
OUTPUTS_DIR = ROOT_DIR / "outputs"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

LABELS_CSV = ANNOTATIONS_DIR / "labels.csv"
NODULE_COORDS_CSV = ANNOTATIONS_DIR / "nodule_coordinates.csv"

CNN_WEIGHTS = MODELS_SAVED_DIR / "lung_cancer_cnn.pth"
UNET_WEIGHTS = MODELS_SAVED_DIR / "unet_segmentation.pth"

# Make sure output/model folders exist so torch.save() etc. don't fail
MODELS_SAVED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Image preprocessing ----------------------------------------------------
IMAGE_SIZE = (224, 224)     # (H, W) for 2D CNN input
HU_MIN, HU_MAX = -1000, 400  # Hounsfield unit clipping window (unused for X-rays)

# --- Training ----------------------------------------------------------------
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
DEVICE = "cuda"  # main.py falls back to "cpu" automatically if unavailable