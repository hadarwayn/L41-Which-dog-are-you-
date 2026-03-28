"""Model evaluation: accuracy, top-5, confusion matrix, classification report."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
)
from tqdm import tqdm

from src.config import DEVICE, TABLES_DIR, ensure_dirs


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: list[str],
    device: torch.device = DEVICE,
    model_name: str = "model",
) -> dict:
    """Comprehensive evaluation of a trained model.

    Returns:
        Dictionary with accuracy, top-5, per-class metrics, confusion matrix.
    """
    ensure_dirs()
    model.eval()
    model = model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc=f"Evaluating {model_name}"):
        images = images.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Metrics
    top1_acc = accuracy_score(all_labels, all_preds)
    top5_acc = top_k_accuracy_score(
        all_labels, all_probs, k=min(5, len(class_names)),
        labels=list(range(len(class_names)))
    )

    conf_matrix = confusion_matrix(all_labels, all_preds)

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "model": model_name,
        "top1_accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
        "num_samples": len(all_labels),
        "num_classes": len(class_names),
        "confusion_matrix": conf_matrix.tolist(),
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
            if name in report
        },
    }

    print(f"\n{model_name} Evaluation:")
    print(f"  Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.1f}%)")
    print(f"  Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.1f}%)")

    # Save results
    save_path = TABLES_DIR / f"{model_name}_evaluation.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def get_predictions_with_confidence(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: list[str],
    device: torch.device = DEVICE,
) -> list[dict]:
    """Get detailed predictions with confidence for every sample.

    Returns:
        List of dicts with: image_idx, true_label, pred_label, confidence,
        top3_breeds, is_correct.
    """
    model.eval()
    model = model.to(device)
    predictions = []

    with torch.no_grad():
        idx = 0
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = torch.softmax(outputs, dim=1)

            for i in range(images.size(0)):
                top3_probs, top3_indices = probs[i].topk(3)
                pred_label = top3_indices[0].item()
                true_label = labels[i].item()

                predictions.append({
                    "idx": idx,
                    "true_label": class_names[true_label],
                    "pred_label": class_names[pred_label],
                    "confidence": top3_probs[0].item(),
                    "is_correct": pred_label == true_label,
                    "top3": [
                        {"breed": class_names[top3_indices[j].item()],
                         "confidence": top3_probs[j].item()}
                        for j in range(3)
                    ],
                })
                idx += 1

    return predictions
