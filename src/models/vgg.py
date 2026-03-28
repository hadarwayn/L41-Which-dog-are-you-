"""VGG-16 (2014) — depth with simplicity, only 3x3 filters.

Key innovations:
- Uniform architecture: only 3x3 convolutions throughout
- Showed that depth matters (16 weight layers)
- Very large: ~138M parameters

Pre-trained on ImageNet.
"""

import torch.nn as nn
from torchvision import models


def get_vgg16(num_classes: int = 120, pretrained: bool = True) -> nn.Module:
    """Load VGG-16 with ImageNet weights, replace classifier for dog breeds.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet weights.

    Returns:
        VGG-16 model with replaced classifier head.
    """
    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weights=weights)

    # Replace final classifier layer: 4096 → num_classes
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model
