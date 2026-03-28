"""Training loop for dog breed classification.

Handles: single epoch training, validation, full training with
early stopping, LR scheduling, checkpointing, and metrics logging.
Works on both CPU (WSL) and GPU (Colab).
"""

import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEVICE, EARLY_STOPPING_PATIENCE,
    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR,
    HISTORY_DIR, MODELS_DIR, ensure_dirs,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = DEVICE,
) -> tuple[float, float]:
    """Train model for one epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # Handle Inception auxiliary outputs
        if isinstance(outputs, tuple):
            loss = criterion(outputs[0], labels)
            if len(outputs) > 1 and outputs[1] is not None:
                loss += 0.4 * criterion(outputs[1], labels)
            logits = outputs[0]
        else:
            loss = criterion(outputs, labels)
            logits = outputs

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> tuple[float, float]:
    """Validate model on validation set.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="  Val", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    class_weights: torch.Tensor = None,
    model_name: str = "model",
    stage_name: str = "",
    device: torch.device = DEVICE,
) -> dict:
    """Full training loop with early stopping and LR scheduling.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        num_epochs: Number of epochs.
        lr: Learning rate.
        class_weights: Optional class weights for CrossEntropyLoss.
        model_name: Model name for logging/saving.
        stage_name: Transfer learning stage name.
        device: Device to train on.

    Returns:
        Training history dictionary.
    """
    ensure_dirs()
    model = model.to(device)

    # Loss function with class weights
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Only optimize parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=LR_SCHEDULER_PATIENCE,
        factor=LR_SCHEDULER_FACTOR,
    )

    history = {
        "model": model_name,
        "stage": stage_name,
        "lr": lr,
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": 0.0,
        "best_epoch": 0,
        "training_time_seconds": 0.0,
    }

    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()

    label = f"{model_name}"
    if stage_name:
        label += f" [{stage_name}]"
    print(f"\n{'='*60}")
    print(f"Training: {label} | LR: {lr} | Epochs: {num_epochs}")
    print(f"Device: {device} | Trainable params: {sum(p.numel() for p in params):,}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch}/{num_epochs} | "
            f"Train: {train_acc:.4f} ({train_loss:.4f}) | "
            f"Val: {val_acc:.4f} ({val_loss:.4f}) | "
            f"LR: {current_lr:.2e} | {epoch_time:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_val_acc"] = best_val_acc
            history["best_epoch"] = epoch
            patience_counter = 0

            save_name = f"best_{model_name}"
            if stage_name:
                save_name += f"_{stage_name}"
            save_path = MODELS_DIR / f"{save_name}.pth"
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={EARLY_STOPPING_PATIENCE})")
            break

    total_time = time.time() - start_time
    history["training_time_seconds"] = total_time

    print(f"  Best val accuracy: {best_val_acc:.4f} (epoch {history['best_epoch']})")
    print(f"  Total training time: {total_time:.1f}s")

    # Save history
    history_path = HISTORY_DIR / f"{model_name}_{stage_name}_history.json" if stage_name \
        else HISTORY_DIR / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history
