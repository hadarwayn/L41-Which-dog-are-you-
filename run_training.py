"""Main training script — trains all models and generates results.

Usage (WSL, 10% subset):
    python run_training.py

Usage (single model):
    python run_training.py --model resnet50

Usage (Colab, full data):
    Set DATA_FRACTION=1.0 in config.py or environment variable.
"""

import argparse
import random
import time

import torch
import numpy as np

from src.config import (
    DEVICE, RANDOM_SEED, MODEL_NAMES, PRETRAINED_MODELS,
    SIMPLE_CNN_LR, SIMPLE_CNN_EPOCHS, DATA_FRACTION,
    ensure_dirs, print_config,
)
from src.models import get_model
from src.data.dataset import get_dataloaders
from src.training.trainer import train_model
from src.training.transfer_learning import train_with_transfer_learning
from src.training.evaluate import evaluate_model
from src.experiments.compare_architectures import compare_architectures
from src.visualization.plots import generate_all_plots


def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_single_model(model_name: str):
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")

    set_seed()

    # Get data
    train_loader, val_loader, class_names, class_weights = get_dataloaders(
        model_name=model_name,
    )

    # Get model
    model = get_model(model_name, num_classes=len(class_names), pretrained=True)

    if model_name in PRETRAINED_MODELS:
        # Transfer learning (3 stages)
        results = train_with_transfer_learning(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            device=DEVICE,
        )
    else:
        # Train from scratch (simple_cnn)
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=SIMPLE_CNN_EPOCHS,
            lr=SIMPLE_CNN_LR,
            class_weights=class_weights,
            model_name=model_name,
            device=DEVICE,
        )

    # Evaluate
    evaluate_model(model, val_loader, class_names, DEVICE, model_name)

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_all_models():
    """Train all 6 models sequentially."""
    print_config()
    ensure_dirs()
    start = time.time()

    for model_name in MODEL_NAMES:
        try:
            train_single_model(model_name)
        except Exception as e:
            print(f"\nERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"ALL TRAINING COMPLETE — {total_time:.1f}s total")
    print(f"{'='*60}")

    # Generate comparison and plots
    compare_architectures()
    generate_all_plots()


def main():
    parser = argparse.ArgumentParser(description="Train dog breed classifiers")
    parser.add_argument("--model", type=str, default=None,
                       choices=MODEL_NAMES + ["all"],
                       help="Model to train (default: all)")
    args = parser.parse_args()

    if args.model and args.model != "all":
        train_single_model(args.model)
    else:
        train_all_models()


if __name__ == "__main__":
    main()
