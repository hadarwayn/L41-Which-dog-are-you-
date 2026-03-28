"""Dataset analysis: class distribution, statistics, and balance reporting."""

from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import TRAIN_DIR, VAL_DIR, GRAPHS_DIR, ensure_dirs


def analyze_dataset(
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    save_plots: bool = True,
) -> dict:
    """Analyze dataset: class distribution, sample counts, imbalance.

    Args:
        train_dir: Path to training directory (with breed subdirs).
        val_dir: Path to validation directory.
        save_plots: Whether to save distribution histogram.

    Returns:
        Dictionary with analysis results.
    """
    ensure_dirs()

    train_counts = _count_per_breed(train_dir)
    val_counts = _count_per_breed(val_dir)

    if not train_counts:
        print("No training data found. Run organize.py first.")
        return {}

    counts = list(train_counts.values())
    breeds = list(train_counts.keys())

    stats = {
        "num_breeds": len(breeds),
        "total_train": sum(counts),
        "total_val": sum(val_counts.values()),
        "min_samples": min(counts),
        "max_samples": max(counts),
        "mean_samples": np.mean(counts),
        "median_samples": np.median(counts),
        "std_samples": np.std(counts),
        "min_breed": breeds[np.argmin(counts)],
        "max_breed": breeds[np.argmax(counts)],
        "breeds_under_50": sum(1 for c in counts if c < 50),
        "breeds_under_30": sum(1 for c in counts if c < 30),
    }

    _print_report(stats, train_counts)

    if save_plots:
        _plot_distribution(train_counts, save_path=GRAPHS_DIR / "class_distribution.png")

    return stats


def _count_per_breed(data_dir: Path) -> dict:
    """Count images per breed directory."""
    counts = {}
    if not data_dir.exists():
        return counts
    for breed_dir in sorted(data_dir.iterdir()):
        if breed_dir.is_dir():
            n = len(list(breed_dir.glob("*.jpg")))
            if n > 0:
                counts[breed_dir.name] = n
    return counts


def _print_report(stats: dict, train_counts: dict):
    """Print formatted analysis report."""
    print("=" * 60)
    print("DATASET ANALYSIS REPORT")
    print("=" * 60)
    print(f"Number of breeds:     {stats['num_breeds']}")
    print(f"Total training imgs:  {stats['total_train']}")
    print(f"Total validation imgs:{stats['total_val']}")
    print(f"Min samples/breed:    {stats['min_samples']} ({stats['min_breed']})")
    print(f"Max samples/breed:    {stats['max_samples']} ({stats['max_breed']})")
    print(f"Mean samples/breed:   {stats['mean_samples']:.1f}")
    print(f"Median samples/breed: {stats['median_samples']:.1f}")
    print(f"Std dev:              {stats['std_samples']:.1f}")
    print(f"Breeds with < 50:     {stats['breeds_under_50']}")
    print(f"Breeds with < 30:     {stats['breeds_under_30']}")

    if stats["breeds_under_50"] > 0:
        print("\nUnderrepresented breeds (< 50 samples):")
        under = {b: c for b, c in train_counts.items() if c < 50}
        for breed, count in sorted(under.items(), key=lambda x: x[1]):
            print(f"  {breed}: {count}")
    print("=" * 60)


def _plot_distribution(
    train_counts: dict,
    save_path: Path = None,
):
    """Plot class distribution histogram."""
    breeds = list(train_counts.keys())
    counts = list(train_counts.values())

    # Sort by count
    sorted_pairs = sorted(zip(breeds, counts), key=lambda x: x[1], reverse=True)
    breeds_sorted = [p[0] for p in sorted_pairs]
    counts_sorted = [p[1] for p in sorted_pairs]

    fig, axes = plt.subplots(2, 1, figsize=(20, 12))

    # Top plot: bar chart of all breeds
    colors = ["#e74c3c" if c < 50 else "#3498db" for c in counts_sorted]
    axes[0].bar(range(len(breeds_sorted)), counts_sorted, color=colors, width=0.8)
    axes[0].set_xlabel("Breed Index (sorted by count)", fontsize=12)
    axes[0].set_ylabel("Number of Images", fontsize=12)
    axes[0].set_title("Class Distribution — Images per Breed", fontsize=14, fontweight="bold")
    axes[0].axhline(y=50, color="red", linestyle="--", alpha=0.7, label="50-sample threshold")
    axes[0].axhline(y=np.mean(counts_sorted), color="green", linestyle="--", alpha=0.7,
                    label=f"Mean ({np.mean(counts_sorted):.0f})")
    axes[0].legend(fontsize=11)

    # Bottom plot: histogram of sample counts
    axes[1].hist(counts_sorted, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Number of Images per Breed", fontsize=12)
    axes[1].set_ylabel("Number of Breeds", fontsize=12)
    axes[1].set_title("Distribution of Breed Sample Sizes", fontsize=14, fontweight="bold")
    axes[1].axvline(x=np.mean(counts_sorted), color="green", linestyle="--",
                    label=f"Mean ({np.mean(counts_sorted):.0f})")
    axes[1].axvline(x=np.median(counts_sorted), color="orange", linestyle="--",
                    label=f"Median ({np.median(counts_sorted):.0f})")
    axes[1].legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Distribution plot saved to {save_path}")

    plt.close()


def compute_class_weights(train_dir: Path = TRAIN_DIR) -> dict:
    """Compute inverse-frequency class weights for balanced training.

    weight[i] = total_samples / (num_classes * count[i])

    Returns:
        Dictionary mapping breed name to weight.
    """
    counts = _count_per_breed(train_dir)
    if not counts:
        return {}

    total = sum(counts.values())
    num_classes = len(counts)

    weights = {}
    for breed, count in counts.items():
        weights[breed] = total / (num_classes * count)

    return weights


if __name__ == "__main__":
    analyze_dataset()
