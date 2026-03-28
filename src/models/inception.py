"""GoogLeNet / Inception v3 (2014) — parallel paths, extreme efficiency.

Key innovations:
- Inception module: parallel 1x1, 3x3, 5x5 convolutions + pooling
- 1x1 convolutions for dimensionality reduction (bottleneck)
- Only ~5M parameters (vs VGG's 138M!)
- Auxiliary classifiers during training

Requires 299x299 input (not 224x224).
Pre-trained on ImageNet.
"""

import torch.nn as nn
from torchvision import models


def get_inception(num_classes: int = 120, pretrained: bool = True) -> nn.Module:
    """Load Inception v3 with ImageNet weights, replace classifier.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet weights.

    Returns:
        Inception v3 model with replaced fc layer.
    """
    weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
    model = models.inception_v3(weights=weights)

    # Replace main classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Replace auxiliary classifier (used during training)
    if model.AuxLogits is not None:
        in_features_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

    return model
