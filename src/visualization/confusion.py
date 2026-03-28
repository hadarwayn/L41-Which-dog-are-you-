"""Confusion matrix visualization for the best model."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import TABLES_DIR, GRAPHS_DIR, ensure_dirs


def plot_confusion_matrix(
    model_name: str = "resnet50",
    top_n: int = 20,
):
    """Plot confusion matrix heatmap for the best model.

    Shows top-N most confused breeds (not full 120x120).

    Args:
        model_name: Model to plot confusion matrix for.
        top_n: Number of breeds to show (most confused).
    """
    ensure_dirs()

    eval_path = TABLES_DIR / f"{model_name}_evaluation.json"
    if not eval_path.exists():
        print(f"No evaluation data for {model_name}. Run evaluate first.")
        return

    with open(eval_path) as f:
        results = json.load(f)

    cm = np.array(results["confusion_matrix"])
    class_names = list(results.get("per_class", {}).keys())

    if not class_names or cm.size == 0:
        print("Empty confusion matrix.")
        return

    # Find most confused breeds (highest off-diagonal values)
    np.fill_diagonal(cm, 0)
    confusion_scores = cm.sum(axis=0) + cm.sum(axis=1)
    top_indices = np.argsort(confusion_scores)[-top_n:]

    # Reload full CM for selected breeds
    with open(eval_path) as f:
        results = json.load(f)
    cm_full = np.array(results["confusion_matrix"])
    cm_subset = cm_full[np.ix_(top_indices, top_indices)]
    names_subset = [class_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_subset,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names_subset,
        yticklabels=names_subset,
        ax=ax,
    )
    ax.set_title(
        f"Confusion Matrix — {model_name} (Top {top_n} Most Confused Breeds)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    save_path = GRAPHS_DIR / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    plot_confusion_matrix()
