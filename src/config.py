"""Central configuration for Dog Breed Classification project.

All paths, hyperparameters, and constants live here.
No hardcoded values in model/training/data code.
"""

import os
import torch
from pathlib import Path


# ==============================================================================
# Environment Detection
# ==============================================================================

def _detect_environment() -> str:
    """Detect if running in Google Colab or WSL/local."""
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


ENVIRONMENT = _detect_environment()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Paths
# ==============================================================================

if ENVIRONMENT == "colab":
    # Google Colab — data on Google Drive
    DRIVE_ROOT = Path("/content/drive/MyDrive/L41_DogBreeds")
    PROJECT_ROOT = Path("/content/L41")
    DATA_ROOT = DRIVE_ROOT / "data"
    RESULTS_ROOT = DRIVE_ROOT / "results"
else:
    # WSL / Local
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    DATA_ROOT = PROJECT_ROOT / "data"
    RESULTS_ROOT = PROJECT_ROOT / "results"

# Data paths
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TRAIN_SUBSET_DIR = PROCESSED_DATA_DIR / "train_subset_10pct"
VAL_SUBSET_DIR = PROCESSED_DATA_DIR / "val_subset_10pct"
TEST_ANIMALS_DIR = PROCESSED_DATA_DIR / "test" / "animals"
TEST_HUMANS_DIR = PROCESSED_DATA_DIR / "test" / "humans"
LABELS_CSV = RAW_DATA_DIR / "labels.csv"

# Results paths
GRAPHS_DIR = RESULTS_ROOT / "graphs"
TABLES_DIR = RESULTS_ROOT / "tables"
EXPERIMENTS_DIR = RESULTS_ROOT / "experiments"
CELEBRITY_GALLERY_DIR = EXPERIMENTS_DIR / "celebrity_gallery"
ANIMAL_GALLERY_DIR = EXPERIMENTS_DIR / "animal_gallery"
MODELS_DIR = RESULTS_ROOT / "models"
HISTORY_DIR = RESULTS_ROOT / "history"

# ==============================================================================
# Dataset
# ==============================================================================

NUM_CLASSES = 120
KAGGLE_DATASET = "dog-breed-identification"
KAGGLE_COMPETITION = "dog-breed-identification"

# Data fraction: 0.1 for WSL local testing, 1.0 for Colab full training
DATA_FRACTION = 0.1 if ENVIRONMENT == "local" else 1.0

# Train/val split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# ==============================================================================
# Image Preprocessing
# ==============================================================================

# Standard input size (VGG, ResNet, AlexNet, MobileNet, Simple CNN)
IMAGE_SIZE = 224

# Inception input size
INCEPTION_IMAGE_SIZE = 299

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==============================================================================
# Training Hyperparameters
# ==============================================================================

BATCH_SIZE = 32
NUM_WORKERS = 2 if ENVIRONMENT == "local" else 4

# Transfer learning stages
TL_STAGES = {
    "stage1": {
        "name": "Feature Extraction (Frozen Backbone)",
        "lr": 1e-3,
        "epochs": 10 if ENVIRONMENT == "colab" else 3,
        "unfreeze_fraction": 0.0,  # All frozen
    },
    "stage2": {
        "name": "Partial Fine-Tuning (Top 25% Unfrozen)",
        "lr": 1e-4,
        "epochs": 10 if ENVIRONMENT == "colab" else 3,
        "unfreeze_fraction": 0.25,
    },
    "stage3": {
        "name": "Full Fine-Tuning (All Unfrozen)",
        "lr": 1e-5,
        "epochs": 10 if ENVIRONMENT == "colab" else 3,
        "unfreeze_fraction": 1.0,
    },
}

# Simple CNN (no transfer learning)
SIMPLE_CNN_LR = 1e-3
SIMPLE_CNN_EPOCHS = 15 if ENVIRONMENT == "colab" else 5

# Early stopping
EARLY_STOPPING_PATIENCE = 5

# LR scheduler
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.1

# ==============================================================================
# Reproducibility
# ==============================================================================

RANDOM_SEED = 42

# ==============================================================================
# Model Names (registry)
# ==============================================================================

MODEL_NAMES = [
    "simple_cnn",
    "alexnet",
    "vgg16",
    "inception",
    "resnet50",
    "mobilenet",
]

# Models that use transfer learning (pretrained weights)
PRETRAINED_MODELS = ["alexnet", "vgg16", "inception", "resnet50", "mobilenet"]

# Models that need 299x299 input
MODELS_299 = ["inception"]

# ==============================================================================
# Experiments
# ==============================================================================

# Non-dog animal types to test (Experiment 2)
ANIMAL_TYPES = [
    "horse", "zebra", "cat", "donkey", "rabbit",
    "fox", "wolf", "bear", "lion", "cow",
]

# Number of celebrity images (Experiment 3)
NUM_CELEBRITIES = 100

# Top-K predictions to record
TOP_K = 3

# ==============================================================================
# Utility
# ==============================================================================

def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_DIR, VAL_DIR,
        TRAIN_SUBSET_DIR, VAL_SUBSET_DIR,
        TEST_ANIMALS_DIR, TEST_HUMANS_DIR,
        GRAPHS_DIR, TABLES_DIR, EXPERIMENTS_DIR,
        CELEBRITY_GALLERY_DIR, ANIMAL_GALLERY_DIR,
        MODELS_DIR, HISTORY_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def print_config():
    """Print current configuration summary."""
    print(f"Environment:    {ENVIRONMENT}")
    print(f"Device:         {DEVICE}")
    print(f"Project root:   {PROJECT_ROOT}")
    print(f"Data root:      {DATA_ROOT}")
    print(f"Data fraction:  {DATA_FRACTION}")
    print(f"Image size:     {IMAGE_SIZE} (Inception: {INCEPTION_IMAGE_SIZE})")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Num classes:    {NUM_CLASSES}")
    print(f"Random seed:    {RANDOM_SEED}")
    print(f"Models:         {', '.join(MODEL_NAMES)}")


if __name__ == "__main__":
    print_config()
