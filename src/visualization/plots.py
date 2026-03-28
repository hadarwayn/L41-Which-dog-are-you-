"""Visualization: accuracy/loss plots, architecture comparison, TL stages."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import HISTORY_DIR, GRAPHS_DIR, MODEL_NAMES, TL_STAGES, ensure_dirs


def plot_accuracy_comparison(history_dir: Path = HISTORY_DIR):
    """Plot accuracy vs epochs for all models (overlaid)."""
    ensure_dirs()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for model_name in MODEL_NAMES:
        epochs, train_acc, val_acc, train_loss, val_loss = _load_best_history(
            model_name, history_dir
        )
        if not epochs:
            continue

        axes[0].plot(epochs, val_acc, label=model_name, linewidth=2)
        axes[1].plot(epochs, val_loss, label=model_name, linewidth=2)

    axes[0].set_title("Validation Accuracy vs Epochs", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Loss vs Epochs", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = GRAPHS_DIR / "accuracy_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

    # Also save loss separately
    _plot_single_metric("loss", history_dir)


def plot_transfer_learning_stages(history_dir: Path = HISTORY_DIR):
    """Bar chart comparing Stage 1/2/3 accuracy for each model."""
    ensure_dirs()
    from src.config import PRETRAINED_MODELS

    models_data = {}
    for model_name in PRETRAINED_MODELS:
        stages_acc = {}
        for stage_key in TL_STAGES:
            path = history_dir / f"{model_name}_{stage_key}_history.json"
            if path.exists():
                with open(path) as f:
                    h = json.load(f)
                stages_acc[stage_key] = h["best_val_acc"]
        if stages_acc:
            models_data[model_name] = stages_acc

    if not models_data:
        print("No transfer learning history found.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models_data))
    width = 0.25
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, stage_key in enumerate(TL_STAGES):
        values = [
            models_data[m].get(stage_key, 0) for m in models_data
        ]
        bars = ax.bar(x + i * width, values, width, label=stage_key, color=colors[i])
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Best Validation Accuracy", fontsize=12)
    ax.set_title("Transfer Learning Stages Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(list(models_data.keys()))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = GRAPHS_DIR / "transfer_learning_stages.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_architecture_bar_chart(history_dir: Path = HISTORY_DIR):
    """Bar chart of final accuracy for all 6 models."""
    ensure_dirs()
    models = []
    accuracies = []

    for model_name in MODEL_NAMES:
        best_acc = _get_best_accuracy(model_name, history_dir)
        if best_acc > 0:
            models.append(model_name)
            accuracies.append(best_acc)

    if not models:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, accuracies, color=colors, edgecolor="white", linewidth=1.5)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f"{acc:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Best Validation Accuracy", fontsize=12)
    ax.set_title("Architecture Comparison — Best Accuracy", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = GRAPHS_DIR / "architecture_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def _load_best_history(model_name: str, history_dir: Path) -> tuple:
    """Load the best training history for a model (across stages)."""
    best_path = None
    best_acc = 0

    # Check stage histories
    for stage in ["stage1", "stage2", "stage3"]:
        path = history_dir / f"{model_name}_{stage}_history.json"
        if path.exists():
            with open(path) as f:
                h = json.load(f)
            if h["best_val_acc"] > best_acc:
                best_acc = h["best_val_acc"]
                best_path = path

    # Check single history
    single = history_dir / f"{model_name}_history.json"
    if single.exists():
        with open(single) as f:
            h = json.load(f)
        if h["best_val_acc"] > best_acc:
            best_path = single

    if best_path is None:
        return [], [], [], [], []

    with open(best_path) as f:
        h = json.load(f)
    return h["epochs"], h["train_acc"], h["val_acc"], h["train_loss"], h["val_loss"]


def _get_best_accuracy(model_name: str, history_dir: Path) -> float:
    """Get the best validation accuracy for a model."""
    best = 0.0
    for path in history_dir.glob(f"{model_name}*_history.json"):
        with open(path) as f:
            h = json.load(f)
        best = max(best, h.get("best_val_acc", 0))
    return best


def _plot_single_metric(metric: str, history_dir: Path):
    """Plot a single metric for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in MODEL_NAMES:
        epochs, train_acc, val_acc, train_loss, val_loss = _load_best_history(
            model_name, history_dir
        )
        if not epochs:
            continue
        values = val_loss if metric == "loss" else val_acc
        ax.plot(epochs, values, label=model_name, linewidth=2)

    title = "Validation Loss" if metric == "loss" else "Validation Accuracy"
    ax.set_title(f"{title} vs Epochs", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = GRAPHS_DIR / f"{metric}_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_plots():
    """Generate all visualization plots."""
    print("\nGenerating plots...")
    plot_accuracy_comparison()
    plot_transfer_learning_stages()
    plot_architecture_bar_chart()
    print("All plots generated.")


if __name__ == "__main__":
    generate_all_plots()
