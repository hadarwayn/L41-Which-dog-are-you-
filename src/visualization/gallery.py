"""Gallery visualizations: celebrity matches, animal matches, example predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from src.config import (
    TABLES_DIR, GRAPHS_DIR, EXPERIMENTS_DIR,
    TEST_HUMANS_DIR, TEST_ANIMALS_DIR, TRAIN_DIR,
    ensure_dirs,
)


def create_celebrity_gallery(top_n: int = 20):
    """Create side-by-side gallery: celebrity → predicted dog breed.

    Args:
        top_n: Number of matches to show.
    """
    ensure_dirs()
    csv_path = TABLES_DIR / "celebrity_dog_matches.csv"
    if not csv_path.exists():
        print("No celebrity results. Run celebrity experiment first.")
        return

    df = pd.read_csv(csv_path)
    df = df.head(top_n)

    n_cols = 4
    n_rows = (len(df) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for i, (_, row) in enumerate(df.iterrows()):
        if i >= len(axes):
            break

        ax = axes[i]

        # Try to load celebrity image
        img_path = TEST_HUMANS_DIR / row["image"]
        if img_path.exists():
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "No Image", ha="center", va="center",
                   transform=ax.transAxes, fontsize=12)

        breed = row["breed_1"].replace("_", " ").title()
        conf = row["conf_1"]
        ax.set_title(
            f"{row['person']}\n→ {breed} ({conf:.1%})",
            fontsize=10, fontweight="bold",
        )
        ax.axis("off")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Celebrity → Dog Breed Matches",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    save_path = GRAPHS_DIR / "celebrity_gallery.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def create_animal_gallery():
    """Create gallery: animal → predicted dog breed."""
    ensure_dirs()
    csv_path = TABLES_DIR / "non_dog_predictions.csv"
    if not csv_path.exists():
        print("No animal results. Run animal experiment first.")
        return

    df = pd.read_csv(csv_path)

    # Pick one image per animal type
    samples = df.groupby("animal_type").first().reset_index()

    n_cols = 5
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for i, (_, row) in enumerate(samples.iterrows()):
        if i >= len(axes):
            break

        ax = axes[i]
        img_path = TEST_ANIMALS_DIR / row["animal_type"] / row["image"]
        if img_path.exists():
            img = mpimg.imread(str(img_path))
            ax.imshow(img)

        breed = row["breed_1"].replace("_", " ").title()
        ax.set_title(
            f"{row['animal_type'].title()}\n→ {breed} ({row['conf_1']:.1%})",
            fontsize=10, fontweight="bold",
        )
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Non-Dog Animals → Predicted Dog Breeds",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    save_path = GRAPHS_DIR / "animal_gallery.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def create_example_predictions_grid(
    predictions: list[dict] = None,
    n_per_category: int = 6,
):
    """Create grid showing correct, wrong, and funny predictions.

    Args:
        predictions: List of prediction dicts (from evaluate.get_predictions_with_confidence).
        n_per_category: Number of examples per category.
    """
    ensure_dirs()

    if predictions is None:
        print("No predictions provided. Pass predictions from evaluate module.")
        return

    correct = [p for p in predictions if p["is_correct"]][:n_per_category]
    wrong = [p for p in predictions if not p["is_correct"]][:n_per_category]

    categories = [
        ("Correct Predictions", correct, "#2ecc71"),
        ("Wrong Predictions", wrong, "#e74c3c"),
    ]

    n_cols = n_per_category
    n_rows = len(categories)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.5))

    for row_idx, (title, preds, color) in enumerate(categories):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]

            if col_idx < len(preds):
                p = preds[col_idx]
                label = f"Pred: {p['pred_label'][:20]}\n"
                if not p["is_correct"]:
                    label += f"True: {p['true_label'][:20]}\n"
                label += f"Conf: {p['confidence']:.1%}"
                ax.text(0.5, 0.5, label, ha="center", va="center",
                       transform=ax.transAxes, fontsize=8,
                       bbox=dict(boxstyle="round", facecolor=color, alpha=0.3))

            ax.set_title(title if col_idx == 0 else "", fontsize=10)
            ax.axis("off")

    plt.suptitle(
        "Example Predictions", fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    save_path = GRAPHS_DIR / "example_predictions.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_confidence_distribution(save_path: Path = None):
    """Histogram of confidence scores: dogs vs animals vs humans."""
    ensure_dirs()

    datasets = {}

    # Dog predictions (from evaluation)
    for eval_file in TABLES_DIR.glob("*_evaluation.json"):
        # Just use the first one found
        import json
        with open(eval_file) as f:
            data = json.load(f)
        # We don't have per-sample confidence from eval JSON,
        # so skip for now — this will be populated from predictions
        break

    # Animal predictions
    animal_csv = TABLES_DIR / "non_dog_predictions.csv"
    if animal_csv.exists():
        df = pd.read_csv(animal_csv)
        datasets["Animals"] = df["conf_1"].values

    # Celebrity predictions
    celeb_csv = TABLES_DIR / "celebrity_dog_matches.csv"
    if celeb_csv.exists():
        df = pd.read_csv(celeb_csv)
        datasets["Celebrities"] = df["conf_1"].values

    if not datasets:
        print("No prediction data for confidence histogram.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Dogs": "#3498db", "Animals": "#e74c3c", "Celebrities": "#f39c12"}

    for label, values in datasets.items():
        ax.hist(values, bins=20, alpha=0.6, label=label,
               color=colors.get(label, "#95a5a6"), edgecolor="white")

    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Confidence Distribution: Dogs vs Animals vs Celebrities",
                fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_path or GRAPHS_DIR / "confidence_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    create_celebrity_gallery()
    create_animal_gallery()
    plot_confidence_distribution()
