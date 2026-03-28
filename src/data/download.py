"""Download and extract the Kaggle Dog Breed Identification dataset.

Works in both WSL (local) and Google Colab environments.
"""

import os
import zipfile
import shutil
from pathlib import Path

from src.config import RAW_DATA_DIR, KAGGLE_COMPETITION, ensure_dirs


def download_dataset(force: bool = False) -> Path:
    """Download the Dog Breed Identification dataset from Kaggle.

    Args:
        force: If True, re-download even if files exist.

    Returns:
        Path to the raw data directory.
    """
    ensure_dirs()
    labels_path = RAW_DATA_DIR / "labels.csv"
    train_dir = RAW_DATA_DIR / "train"

    if labels_path.exists() and train_dir.exists() and not force:
        print(f"Dataset already exists at {RAW_DATA_DIR}")
        _print_stats(train_dir, labels_path)
        return RAW_DATA_DIR

    print("Downloading Dog Breed Identification dataset from Kaggle...")
    print("Make sure your Kaggle API key is configured:")
    print("  - Place kaggle.json in ~/.kaggle/kaggle.json")
    print("  - Or set KAGGLE_USERNAME and KAGGLE_KEY env vars")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(
            KAGGLE_COMPETITION,
            path=str(RAW_DATA_DIR),
        )
        print("Download complete. Extracting...")
        _extract_all(RAW_DATA_DIR)
        _print_stats(train_dir, labels_path)
        return RAW_DATA_DIR

    except Exception as e:
        print(f"Kaggle API download failed: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/c/dog-breed-identification/data")
        print("2. Download train.zip, test.zip, and labels.csv")
        print(f"3. Place them in: {RAW_DATA_DIR}")
        print("4. Run this script again")
        _try_extract_existing(RAW_DATA_DIR)
        return RAW_DATA_DIR


def _extract_all(data_dir: Path):
    """Extract all zip files in the data directory."""
    for zip_path in data_dir.glob("*.zip"):
        print(f"  Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        zip_path.unlink()
        print(f"  Deleted {zip_path.name}")


def _try_extract_existing(data_dir: Path):
    """Try to extract any existing zip files."""
    zips = list(data_dir.glob("*.zip"))
    if zips:
        print(f"Found {len(zips)} zip file(s), extracting...")
        _extract_all(data_dir)


def _print_stats(train_dir: Path, labels_path: Path):
    """Print basic dataset statistics."""
    if train_dir.exists():
        images = list(train_dir.glob("*.jpg"))
        print(f"Training images: {len(images)}")

    if labels_path.exists():
        import pandas as pd
        df = pd.read_csv(labels_path)
        print(f"Labels entries: {len(df)}")
        print(f"Unique breeds: {df['breed'].nunique()}")


if __name__ == "__main__":
    download_dataset()
