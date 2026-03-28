"""PyTorch Dataset and DataLoader setup for dog breed classification.

Supports:
- Full dataset (Colab) and 10% subset (WSL)
- Class-weighted sampling for imbalance
- Both 224x224 and 299x299 input sizes
"""

from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from src.config import (
    TRAIN_DIR, VAL_DIR, TRAIN_SUBSET_DIR, VAL_SUBSET_DIR,
    BATCH_SIZE, NUM_WORKERS, DATA_FRACTION, RANDOM_SEED,
)
from src.data.augmentation import get_transforms_for_model
from src.data.analysis import compute_class_weights


def get_dataloaders(
    model_name: str = "resnet50",
    batch_size: int = BATCH_SIZE,
    use_subset: bool = None,
    use_weighted_sampler: bool = True,
) -> tuple:
    """Create training and validation DataLoaders.

    Args:
        model_name: Model name (determines input size: 224 or 299).
        batch_size: Batch size.
        use_subset: If True, use 10% subset. Auto-detected from DATA_FRACTION.
        use_weighted_sampler: If True, use WeightedRandomSampler for class balance.

    Returns:
        Tuple of (train_loader, val_loader, class_names, class_weights_tensor).
    """
    if use_subset is None:
        use_subset = DATA_FRACTION < 1.0

    train_dir = TRAIN_SUBSET_DIR if use_subset else TRAIN_DIR
    val_dir = VAL_SUBSET_DIR if use_subset else VAL_DIR

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_dir}. "
            "Run organize.py first."
        )

    # Get transforms
    train_transform = get_transforms_for_model(model_name, is_train=True)
    val_transform = get_transforms_for_model(model_name, is_train=False)

    # Create datasets
    train_dataset = ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = ImageFolder(str(val_dir), transform=val_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, "
          f"{num_classes} classes")

    # Compute class weights
    class_weights_tensor = _compute_weight_tensor(train_dataset, num_classes)

    # Create sampler for balanced training
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sampler = _create_weighted_sampler(train_dataset, class_weights_tensor)
        shuffle = False  # Sampler handles this

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, class_names, class_weights_tensor


def _compute_weight_tensor(dataset: ImageFolder, num_classes: int) -> torch.Tensor:
    """Compute class weight tensor for CrossEntropyLoss.

    weight[i] = total / (num_classes * count[i])
    """
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets, minlength=num_classes).astype(float)
    class_counts = np.maximum(class_counts, 1.0)  # Avoid division by zero
    total = len(targets)

    weights = total / (num_classes * class_counts)
    return torch.FloatTensor(weights)


def _create_weighted_sampler(
    dataset: ImageFolder,
    class_weights: torch.Tensor,
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for balanced batch sampling."""
    sample_weights = [class_weights[label].item() for _, label in dataset.samples]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_breed_names(train_dir: Path = TRAIN_DIR) -> list[str]:
    """Get sorted list of breed names from directory structure."""
    if not train_dir.exists():
        return []
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


if __name__ == "__main__":
    train_loader, val_loader, classes, weights = get_dataloaders()
    print(f"\nClasses: {len(classes)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Show one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
