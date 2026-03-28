"""AlexNet (2012) — the architecture that started the deep learning revolution.

Key innovations:
- ReLU activation (instead of sigmoid/tanh)
- Dropout regularization
- GPU training
- Large filters: 11x11, 5x5 in early layers

~60M parameters. Pre-trained on ImageNet.
"""

import torch.nn as nn
from torchvision import models


def get_alexnet(num_classes: int = 120, pretrained: bool = True) -> nn.Module:
    """Load AlexNet with ImageNet weights, replace classifier for dog breeds.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet weights.

    Returns:
        AlexNet model with replaced classifier head.
    """
    weights = models.AlexNet_Weights.DEFAULT if pretrained else None
    model = models.alexnet(weights=weights)

    # Replace final classifier layer: 4096 → num_classes
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model
