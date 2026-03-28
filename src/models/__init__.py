"""Model registry — get any architecture by name.

Usage:
    from src.models import get_model
    model = get_model("resnet50", num_classes=120, pretrained=True)
"""

import torch.nn as nn

from src.models.simple_cnn import get_simple_cnn
from src.models.alexnet import get_alexnet
from src.models.vgg import get_vgg16
from src.models.inception import get_inception
from src.models.resnet import get_resnet50
from src.models.mobilenet import get_mobilenet


_MODEL_REGISTRY = {
    "simple_cnn": get_simple_cnn,
    "alexnet": get_alexnet,
    "vgg16": get_vgg16,
    "inception": get_inception,
    "resnet50": get_resnet50,
    "mobilenet": get_mobilenet,
}


def get_model(
    name: str,
    num_classes: int = 120,
    pretrained: bool = True,
) -> nn.Module:
    """Get a model by name from the registry.

    Args:
        name: One of: simple_cnn, alexnet, vgg16, inception, resnet50, mobilenet.
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        PyTorch model ready for training.

    Raises:
        ValueError: If model name is not in registry.
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    kwargs = {"num_classes": num_classes}
    if name != "simple_cnn":
        kwargs["pretrained"] = pretrained

    model = _MODEL_REGISTRY[name](**kwargs)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {name}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    return model


def list_models() -> list[str]:
    """Return list of available model names."""
    return list(_MODEL_REGISTRY.keys())
