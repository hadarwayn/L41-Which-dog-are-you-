# Python & Machine Learning Code Standards

## Python Conventions

### Naming
- `snake_case` for files, functions, variables
- `PascalCase` for classes
- `UPPER_SNAKE` for constants
- Prefix private methods with `_`

### Type Hints
```python
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
```

### File Organization
- Max 200 lines per file
- Max 50 lines per function
- One class per file (for models)
- Group imports: stdlib → third-party → local

### Docstrings
```python
def get_model(name: str, num_classes: int = 120, pretrained: bool = True) -> nn.Module:
    """Load a model architecture by name.

    Args:
        name: One of 'simple_cnn', 'alexnet', 'vgg16', 'inception', 'resnet50', 'mobilenet'
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet weights

    Returns:
        PyTorch model ready for training
    """
```

## ML-Specific Standards

### Reproducibility
```python
# ALWAYS set seeds at the start of training
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```

### Device Handling
```python
# Auto-detect device — works on both WSL (CPU) and Colab (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Configuration
- ALL hyperparameters in `src/config.py`
- ALL paths in `src/config.py`
- NO hardcoded values in model/training code
- Support `DATA_FRACTION` param (0.1 for WSL, 1.0 for Colab)

### Data Pipeline
- Use `torchvision.datasets.ImageFolder` for folder-based datasets
- Use `torch.utils.data.DataLoader` with `num_workers` > 0
- Apply augmentation ONLY to training set
- Normalize with ImageNet mean/std
- Validate pipeline with sample visualization before training

### Training Loop
```python
# Always include:
model.train()                    # Set training mode
with torch.no_grad():            # During evaluation
model.eval()                     # Set eval mode
torch.save(model.state_dict())   # Checkpoint
```

### Metrics Logging
- Log loss and accuracy per epoch as JSON
- Track training time per model
- Save confusion matrix data
- Record all hyperparameters used

### Memory Management
```python
# Clear GPU cache between models
torch.cuda.empty_cache()

# Use mixed precision for large models (VGG-16)
from torch.cuda.amp import autocast, GradScaler
```

## Anti-Patterns (DO NOT)

- DO NOT hardcode file paths — use `config.py`
- DO NOT skip `model.eval()` during validation
- DO NOT forget `optimizer.zero_grad()` in training loop
- DO NOT load entire dataset into memory — use DataLoader
- DO NOT train on validation data
- DO NOT use `torch.no_grad()` during training
- DO NOT ignore class imbalance — always use weighted loss or sampling
