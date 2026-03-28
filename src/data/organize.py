"""Organize raw Kaggle dataset into breed-folder structure.

Reads labels.csv, creates train/{breed}/ and val/{breed}/ directories,
copies images with stratified split, and standardizes resolution.
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image

from src.config import (
    RAW_DATA_DIR, TRAIN_DIR, VAL_DIR, LABELS_CSV,
    TRAIN_SPLIT, IMAGE_SIZE, INCEPTION_IMAGE_SIZE,
    RANDOM_SEED, ensure_dirs,
)


def organize_dataset(force: bool = False) -> dict:
    """Organize raw images into breed subdirectories with train/val split.

    Creates:
        data/processed/train/{breed}/image.jpg
        data/processed/val/{breed}/image.jpg

    Args:
        force: If True, re-organize even if directories exist.

    Returns:
        Dictionary with organization statistics.
    """
    ensure_dirs()

    if _is_organized() and not force:
        print("Dataset already organized. Use force=True to re-organize.")
        return _get_stats()

    print("Organizing dataset into breed folders...")

    if not LABELS_CSV.exists():
        raise FileNotFoundError(
            f"labels.csv not found at {LABELS_CSV}. "
            "Run download.py first."
        )

    df = pd.read_csv(LABELS_CSV)
    raw_train_dir = RAW_DATA_DIR / "train"

    if not raw_train_dir.exists():
        raise FileNotFoundError(
            f"Raw training images not found at {raw_train_dir}. "
            "Run download.py first."
        )

    # Clean existing processed dirs
    for d in [TRAIN_DIR, VAL_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # Group images by breed
    breed_images = defaultdict(list)
    for _, row in df.iterrows():
        img_path = raw_train_dir / f"{row['id']}.jpg"
        if img_path.exists():
            breed_images[row["breed"]].append(img_path)

    # Stratified split
    random.seed(RANDOM_SEED)
    stats = {"breeds": 0, "train": 0, "val": 0, "skipped": 0}

    for breed, images in sorted(breed_images.items()):
        random.shuffle(images)
        split_idx = max(1, int(len(images) * TRAIN_SPLIT))
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Create breed directories
        train_breed_dir = TRAIN_DIR / breed
        val_breed_dir = VAL_DIR / breed
        train_breed_dir.mkdir(parents=True, exist_ok=True)
        val_breed_dir.mkdir(parents=True, exist_ok=True)

        # Copy and resize images
        for img_path in train_imgs:
            if _copy_and_resize(img_path, train_breed_dir / img_path.name):
                stats["train"] += 1
            else:
                stats["skipped"] += 1

        for img_path in val_imgs:
            if _copy_and_resize(img_path, val_breed_dir / img_path.name):
                stats["val"] += 1
            else:
                stats["skipped"] += 1

        stats["breeds"] += 1

    print(f"Organization complete:")
    print(f"  Breeds:  {stats['breeds']}")
    print(f"  Train:   {stats['train']} images")
    print(f"  Val:     {stats['val']} images")
    print(f"  Skipped: {stats['skipped']} (corrupted/unreadable)")
    return stats


def _copy_and_resize(src: Path, dst: Path, size: int = IMAGE_SIZE) -> bool:
    """Copy image to destination, resizing to standard resolution.

    Args:
        src: Source image path.
        dst: Destination image path.
        size: Target size (square).

    Returns:
        True if successful, False if image is corrupted.
    """
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(dst, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"  Warning: Could not process {src.name}: {e}")
        return False


def _is_organized() -> bool:
    """Check if dataset is already organized."""
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        return False
    train_breeds = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    return len(train_breeds) > 50


def _get_stats() -> dict:
    """Get statistics of already organized dataset."""
    train_breeds = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    train_count = sum(len(list(d.glob("*.jpg"))) for d in train_breeds)
    val_breeds = [d for d in VAL_DIR.iterdir() if d.is_dir()]
    val_count = sum(len(list(d.glob("*.jpg"))) for d in val_breeds)
    print(f"Existing organization: {len(train_breeds)} breeds, "
          f"{train_count} train, {val_count} val")
    return {
        "breeds": len(train_breeds),
        "train": train_count,
        "val": val_count,
        "skipped": 0,
    }


def create_stratified_subset(
    fraction: float = 0.1,
    src_train: Path = TRAIN_DIR,
    src_val: Path = VAL_DIR,
) -> dict:
    """Create a stratified subset maintaining breed distribution.

    Args:
        fraction: Fraction of data to keep (e.g., 0.1 for 10%).
        src_train: Source training directory.
        src_val: Source validation directory.

    Returns:
        Dictionary with subset statistics.
    """
    from src.config import TRAIN_SUBSET_DIR, VAL_SUBSET_DIR

    print(f"Creating {fraction*100:.0f}% stratified subset...")
    random.seed(RANDOM_SEED)

    stats = {"breeds": 0, "train": 0, "val": 0}

    for src, dst in [(src_train, TRAIN_SUBSET_DIR), (src_val, VAL_SUBSET_DIR)]:
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

        split_name = "train" if "train" in str(dst) else "val"

        for breed_dir in sorted(src.iterdir()):
            if not breed_dir.is_dir():
                continue
            images = sorted(breed_dir.glob("*.jpg"))
            n_keep = max(1, int(len(images) * fraction))
            selected = random.sample(images, n_keep)

            subset_breed_dir = dst / breed_dir.name
            subset_breed_dir.mkdir(parents=True, exist_ok=True)

            for img in selected:
                shutil.copy2(img, subset_breed_dir / img.name)
                stats[split_name] += 1

            if split_name == "train":
                stats["breeds"] += 1

    print(f"Subset created:")
    print(f"  Breeds: {stats['breeds']}")
    print(f"  Train:  {stats['train']} images")
    print(f"  Val:    {stats['val']} images")
    return stats


if __name__ == "__main__":
    organize_dataset()
    create_stratified_subset(fraction=0.1)
