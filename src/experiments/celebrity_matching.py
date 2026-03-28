"""Experiment 3: Match 100 famous people to dog breeds.

For each celebrity:
  1. Load face image
  2. Resize to 224x224, normalize
  3. Feed through best trained model
  4. Softmax → top-3 dog breeds + confidence
  5. Create side-by-side gallery
"""

from pathlib import Path

import pandas as pd

from src.config import (
    TEST_HUMANS_DIR, TABLES_DIR, TOP_K, DEVICE, ensure_dirs,
)
from src.inference.predict import predict_dog_breed, load_trained_model
from src.data.dataset import get_breed_names


def run_celebrity_experiment(
    model_name: str = "resnet50",
    humans_dir: Path = TEST_HUMANS_DIR,
) -> pd.DataFrame:
    """Run celebrity images through the trained dog breed model.

    Args:
        model_name: Which trained model to use.
        humans_dir: Directory with celebrity images.

    Returns:
        DataFrame with predictions for each celebrity.
    """
    ensure_dirs()

    print(f"\n{'='*60}")
    print("EXPERIMENT 3: Celebrity → Dog Breed Matching")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    model = load_trained_model(model_name)
    breed_names = get_breed_names()

    images = (
        list(humans_dir.glob("*.jpg"))
        + list(humans_dir.glob("*.jpeg"))
        + list(humans_dir.glob("*.png"))
    )

    if not images:
        print(f"No celebrity images found in {humans_dir}")
        print("Add images named like: elon_musk.jpg, beyonce.jpg, etc.")
        return pd.DataFrame()

    print(f"Found {len(images)} celebrity images")
    rows = []

    for img_path in sorted(images):
        # Extract person name from filename
        person_name = img_path.stem.replace("_", " ").replace("-", " ").title()

        preds = predict_dog_breed(
            str(img_path), model, breed_names, TOP_K, DEVICE
        )

        row = {
            "person": person_name,
            "image": img_path.name,
            "breed_1": preds[0]["breed"] if len(preds) > 0 else "",
            "conf_1": preds[0]["confidence"] if len(preds) > 0 else 0,
            "breed_2": preds[1]["breed"] if len(preds) > 1 else "",
            "conf_2": preds[1]["confidence"] if len(preds) > 1 else 0,
            "breed_3": preds[2]["breed"] if len(preds) > 2 else "",
            "conf_3": preds[2]["confidence"] if len(preds) > 2 else 0,
        }
        rows.append(row)
        print(f"  {person_name}: {row['breed_1']} ({row['conf_1']:.1%})")

    df = pd.DataFrame(rows)
    if not df.empty:
        csv_path = TABLES_DIR / "celebrity_dog_matches.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        _print_analysis(df)

    return df


def _print_analysis(df: pd.DataFrame):
    """Print fun analysis of celebrity results."""
    print(f"\n{'='*60}")
    print("FUN ANALYSIS")
    print(f"{'='*60}")
    print(f"Celebrities analyzed: {len(df)}")
    print(f"Average confidence: {df['conf_1'].mean():.1%}")

    print("\nMost common 'celebrity breed':")
    breed_counts = df["breed_1"].value_counts().head(10)
    for breed, count in breed_counts.items():
        people = df[df["breed_1"] == breed]["person"].tolist()
        print(f"  {breed}: {count} people ({', '.join(people[:3])}...)")

    print("\nHighest confidence matches (most 'dog-like' celebrities):")
    top = df.nlargest(5, "conf_1")
    for _, row in top.iterrows():
        print(f"  {row['person']} → {row['breed_1']} ({row['conf_1']:.1%})")

    print("\nLowest confidence (least 'dog-like'):")
    bottom = df.nsmallest(5, "conf_1")
    for _, row in bottom.iterrows():
        print(f"  {row['person']} → {row['breed_1']} ({row['conf_1']:.1%})")


if __name__ == "__main__":
    run_celebrity_experiment()
