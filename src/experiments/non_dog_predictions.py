"""Experiment 2: Run 10 non-dog animal types through the trained NN.

Animals: horse, zebra, cat, donkey, rabbit, fox, wolf, bear, lion, cow
For each: preprocess → model → softmax → top-3 dog breeds + confidence.
"""

from pathlib import Path

import pandas as pd

from src.config import (
    TEST_ANIMALS_DIR, TABLES_DIR, ANIMAL_TYPES,
    TOP_K, DEVICE, ensure_dirs,
)
from src.inference.predict import predict_dog_breed, load_trained_model
from src.data.dataset import get_breed_names


def run_animal_experiment(
    model_name: str = "resnet50",
    animals_dir: Path = TEST_ANIMALS_DIR,
) -> pd.DataFrame:
    """Run non-dog animal images through the trained model.

    Args:
        model_name: Which trained model to use.
        animals_dir: Directory with animal type subdirectories.

    Returns:
        DataFrame with predictions for each animal image.
    """
    ensure_dirs()

    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Non-Dog Animal Predictions")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    model = load_trained_model(model_name)
    breed_names = get_breed_names()
    rows = []

    for animal_type in ANIMAL_TYPES:
        type_dir = animals_dir / animal_type
        if not type_dir.exists():
            print(f"  Skipping {animal_type} (no images at {type_dir})")
            continue

        images = list(type_dir.glob("*.jpg")) + list(type_dir.glob("*.png"))
        if not images:
            print(f"  Skipping {animal_type} (no images found)")
            continue

        print(f"\n  {animal_type.upper()} ({len(images)} images):")

        for img_path in images:
            preds = predict_dog_breed(
                str(img_path), model, breed_names, TOP_K, DEVICE
            )
            row = {
                "animal_type": animal_type,
                "image": img_path.name,
                "breed_1": preds[0]["breed"] if len(preds) > 0 else "",
                "conf_1": preds[0]["confidence"] if len(preds) > 0 else 0,
                "breed_2": preds[1]["breed"] if len(preds) > 1 else "",
                "conf_2": preds[1]["confidence"] if len(preds) > 1 else 0,
                "breed_3": preds[2]["breed"] if len(preds) > 2 else "",
                "conf_3": preds[2]["confidence"] if len(preds) > 2 else 0,
            }
            rows.append(row)
            print(f"    {img_path.name}: {row['breed_1']} ({row['conf_1']:.1%})")

    df = pd.DataFrame(rows)
    if not df.empty:
        csv_path = TABLES_DIR / "non_dog_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        _print_analysis(df)

    return df


def _print_analysis(df: pd.DataFrame):
    """Print analysis of animal prediction results."""
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    print(f"Total images analyzed: {len(df)}")
    print(f"Average confidence (top-1): {df['conf_1'].mean():.1%}")
    print(f"Max confidence: {df['conf_1'].max():.1%}")
    print(f"Min confidence: {df['conf_1'].min():.1%}")

    print("\nMost predicted breeds across all animals:")
    breed_counts = df["breed_1"].value_counts().head(10)
    for breed, count in breed_counts.items():
        print(f"  {breed}: {count} times")

    print("\nPer animal type (top prediction):")
    for animal in df["animal_type"].unique():
        subset = df[df["animal_type"] == animal]
        top_breed = subset["breed_1"].mode().iloc[0] if not subset.empty else "?"
        avg_conf = subset["conf_1"].mean()
        print(f"  {animal}: {top_breed} (avg conf: {avg_conf:.1%})")


if __name__ == "__main__":
    run_animal_experiment()
