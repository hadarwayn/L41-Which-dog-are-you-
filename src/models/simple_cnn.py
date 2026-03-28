"""Simple CNN baseline — built from scratch, no pretrained weights.

Architecture:
  3x [Conv2D → BatchNorm → ReLU → MaxPool] → Flatten → Dense(512) → Dropout → Dense(120)

Demonstrates core CNN building blocks: convolution, pooling, activation,
batch normalization, dropout, fully connected layers, and softmax output.
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """Custom baseline CNN for dog breed classification.

    3 convolutional blocks followed by 2 fully connected layers.
    Uses all fundamental CNN operations.
    """

    def __init__(self, num_classes: int = 120):
        super().__init__()

        # Block 1: 3x224x224 → 32x112x112
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 32x112x112 → 64x56x56
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 64x56x56 → 128x28x28
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Average Pooling → 128x1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def get_simple_cnn(num_classes: int = 120, **kwargs) -> SimpleCNN:
    """Factory function for Simple CNN."""
    return SimpleCNN(num_classes=num_classes)
