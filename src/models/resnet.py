"""ResNet-50 (2015) — skip connections solve vanishing gradients.

Key innovations:
- Residual blocks with skip connections: H(x) = F(x) + x
- Bottleneck design: 1x1 → 3x3 → 1x1 convolutions
- Can train 50+ layers without vanishing gradients
- ~25M parameters

Pre-trained on ImageNet.
"""

import torch.nn as nn
from torchvision import models


def get_resnet50(num_classes: int = 120, pretrained: bool = True) -> nn.Module:
    """Load ResNet-50 with ImageNet weights, replace classifier.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet weights.

    Returns:
        ResNet-50 model with replaced fc layer.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Replace final fully connected layer: 2048 → num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
