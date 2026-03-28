"""3-stage transfer learning pipeline.

Stage 1: Feature Extraction — freeze backbone, train classifier only
Stage 2: Partial Fine-Tuning — unfreeze top 25% of backbone
Stage 3: Full Fine-Tuning — unfreeze entire network
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import DEVICE, TL_STAGES, PRETRAINED_MODELS, MODELS_DIR
from src.training.trainer import train_model


def freeze_backbone(model: nn.Module) -> int:
    """Freeze all parameters in the model.

    Returns:
        Number of frozen parameters.
    """
    frozen = 0
    for param in model.parameters():
        param.requires_grad = False
        frozen += param.numel()
    return frozen


def unfreeze_top_fraction(model: nn.Module, fraction: float) -> int:
    """Unfreeze the top fraction of model parameters.

    Args:
        fraction: 0.0 = all frozen, 1.0 = all unfrozen, 0.25 = top 25%.

    Returns:
        Number of unfrozen parameters.
    """
    all_params = list(model.parameters())
    n_total = len(all_params)

    if fraction >= 1.0:
        for param in all_params:
            param.requires_grad = True
        return sum(p.numel() for p in all_params)

    if fraction <= 0.0:
        for param in all_params:
            param.requires_grad = False
        return 0

    # Unfreeze from the end (top layers)
    n_unfreeze = max(1, int(n_total * fraction))
    cutoff = n_total - n_unfreeze

    unfrozen = 0
    for i, param in enumerate(all_params):
        if i >= cutoff:
            param.requires_grad = True
            unfrozen += param.numel()
        else:
            param.requires_grad = False

    return unfrozen


def _unfreeze_classifier(model: nn.Module, model_name: str):
    """Unfreeze only the classifier head (always trainable)."""
    classifier_attrs = {
        "alexnet": "classifier",
        "vgg16": "classifier",
        "inception": "fc",
        "resnet50": "fc",
        "mobilenet": "classifier",
    }

    attr = classifier_attrs.get(model_name)
    if attr is None:
        return

    classifier = getattr(model, attr, None)
    if classifier is not None:
        for param in classifier.parameters():
            param.requires_grad = True


def train_with_transfer_learning(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor = None,
    device: torch.device = DEVICE,
) -> dict:
    """Run 3-stage transfer learning pipeline.

    Args:
        model: Pre-trained model with replaced classifier.
        model_name: Model name for logging.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        class_weights: Class weights for balanced loss.
        device: Device to train on.

    Returns:
        Dictionary with results from all 3 stages.
    """
    results = {"model": model_name, "stages": {}}

    for stage_key, stage_cfg in TL_STAGES.items():
        stage_name = stage_cfg["name"]
        lr = stage_cfg["lr"]
        epochs = stage_cfg["epochs"]
        fraction = stage_cfg["unfreeze_fraction"]

        print(f"\n{'#'*60}")
        print(f"  {stage_key.upper()}: {stage_name}")
        print(f"{'#'*60}")

        # Freeze/unfreeze
        if fraction == 0.0:
            freeze_backbone(model)
            _unfreeze_classifier(model, model_name)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Backbone frozen. Trainable params: {trainable:,}")
        else:
            unfrozen = unfreeze_top_fraction(model, fraction)
            print(f"  Unfrozen {fraction*100:.0f}% ({unfrozen:,} params)")

        # Train
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            lr=lr,
            class_weights=class_weights,
            model_name=model_name,
            stage_name=stage_key,
            device=device,
        )

        results["stages"][stage_key] = {
            "name": stage_name,
            "best_val_acc": history["best_val_acc"],
            "best_epoch": history["best_epoch"],
            "training_time": history["training_time_seconds"],
            "final_train_acc": history["train_acc"][-1] if history["train_acc"] else 0,
            "final_val_acc": history["val_acc"][-1] if history["val_acc"] else 0,
        }

    # Determine best stage
    best_stage = max(
        results["stages"].items(),
        key=lambda x: x[1]["best_val_acc"]
    )
    results["best_stage"] = best_stage[0]
    results["best_accuracy"] = best_stage[1]["best_val_acc"]

    # Load best weights from best stage
    best_path = MODELS_DIR / f"best_{model_name}_{best_stage[0]}.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        # Also save as overall best
        overall_best = MODELS_DIR / f"best_{model_name}.pth"
        torch.save(model.state_dict(), overall_best)

    print(f"\n{'='*60}")
    print(f"Transfer Learning Summary for {model_name}:")
    for sk, sv in results["stages"].items():
        marker = " <<<" if sk == results["best_stage"] else ""
        print(f"  {sk}: {sv['best_val_acc']:.4f} acc, "
              f"{sv['training_time']:.1f}s{marker}")
    print(f"  Best: {results['best_stage']} ({results['best_accuracy']:.4f})")
    print(f"{'='*60}")

    return results
