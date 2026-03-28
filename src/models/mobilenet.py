"""MobileNet v2 (2018) — efficient architecture for mobile devices.

Key innovations:
- Depthwise separable convolutions (much fewer params)
- Inverted residuals with linear bottlenecks
- Only ~3.4M parameters
- Fast inference — designed for phones and edge devices

Pre-trained on ImageNet.
"""

import torch.nn as nn
from torchvision import models


def get_mobilenet(num_classes: int = 120, pretrained: bool = True) -> nn.Module:
    """Load MobileNet v2 with ImageNet weights, replace classifier.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet weights.

    Returns:
        MobileNet v2 model with replaced classifier head.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Replace final classifier: 1280 → num_classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
