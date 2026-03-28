"""Data augmentation and preprocessing transforms.

Provides separate transform pipelines for:
- Training (with augmentation)
- Validation/test (resize + normalize only)
- Both 224x224 (most models) and 299x299 (Inception)
"""

from torchvision import transforms

from src.config import IMAGE_SIZE, INCEPTION_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Training transforms with data augmentation.

    Args:
        image_size: Target image size (224 or 299).

    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Validation/test transforms (no augmentation).

    Args:
        image_size: Target image size (224 or 299).

    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # Slight upscale for center crop
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Inference transforms for arbitrary images (experiments).

    Same as validation — no augmentation, just resize + normalize.
    """
    return get_val_transforms(image_size)


def get_transforms_for_model(model_name: str, is_train: bool) -> transforms.Compose:
    """Get appropriate transforms based on model name and mode.

    Args:
        model_name: One of the model names from config.MODEL_NAMES.
        is_train: True for training, False for validation/inference.

    Returns:
        Composed transform pipeline.
    """
    from src.config import MODELS_299

    size = INCEPTION_IMAGE_SIZE if model_name in MODELS_299 else IMAGE_SIZE

    if is_train:
        return get_train_transforms(size)
    return get_val_transforms(size)


def denormalize(tensor):
    """Reverse ImageNet normalization for visualization.

    Args:
        tensor: Normalized image tensor (C, H, W).

    Returns:
        Denormalized tensor with values in [0, 1].
    """
    import torch

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
