"""Experiment 1: Compare all 6 architectures on dog breed classification.

Loads training histories and generates comparison tables and plots.
"""

import json
from pathlib import Path

import pandas as pd

from src.config import HISTORY_DIR, TABLES_DIR, MODELS_DIR, MODEL_NAMES, ensure_dirs


def compare_architectures(history_dir: Path = HISTORY_DIR) -> pd.DataFrame:
    """Aggregate results from all trained models into comparison table.

    Returns:
        DataFrame with: Model | Top-1 Acc | Top-5 Acc | Params | Time | Best Stage
    """
    ensure_dirs()
    rows = []

    for model_name in MODEL_NAMES:
        # Look for stage histories or single history
        best_acc = 0.0
        best_stage = "N/A"
        total_time = 0.0
        top5_acc = None

        # Check for transfer learning stages
        for stage in ["stage1", "stage2", "stage3"]:
            path = history_dir / f"{model_name}_{stage}_history.json"
            if path.exists():
                with open(path) as f:
                    h = json.load(f)
                if h["best_val_acc"] > best_acc:
                    best_acc = h["best_val_acc"]
                    best_stage = stage
                total_time += h.get("training_time_seconds", 0)

        # Check for single history (simple_cnn)
        single_path = history_dir / f"{model_name}_history.json"
        if single_path.exists() and best_acc == 0:
            with open(single_path) as f:
                h = json.load(f)
            best_acc = h["best_val_acc"]
            total_time = h.get("training_time_seconds", 0)

        # Check for evaluation results (has top-5)
        eval_path = TABLES_DIR / f"{model_name}_evaluation.json"
        if eval_path.exists():
            with open(eval_path) as f:
                ev = json.load(f)
            top5_acc = ev.get("top5_accuracy")
            if ev.get("top1_accuracy", 0) > best_acc:
                best_acc = ev["top1_accuracy"]

        # Count params from model file
        params = _count_model_params(model_name)

        if best_acc > 0 or params > 0:
            rows.append({
                "Model": model_name,
                "Top-1 Acc": f"{best_acc:.4f}" if best_acc > 0 else "—",
                "Top-5 Acc": f"{top5_acc:.4f}" if top5_acc else "—",
                "Params": f"{params:,}" if params > 0 else "—",
                "Train Time (s)": f"{total_time:.1f}" if total_time > 0 else "—",
                "Best Stage": best_stage,
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        print("\n" + "=" * 80)
        print("ARCHITECTURE COMPARISON")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        # Save to CSV
        csv_path = TABLES_DIR / "model_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")
    else:
        print("No training results found. Train models first.")

    return df


def _count_model_params(model_name: str) -> int:
    """Count total parameters for a model."""
    try:
        from src.models import get_model
        model = get_model(model_name, pretrained=False)
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return 0


def get_all_histories(history_dir: Path = HISTORY_DIR) -> dict:
    """Load all training histories into a dictionary."""
    histories = {}
    for path in history_dir.glob("*_history.json"):
        with open(path) as f:
            h = json.load(f)
        key = path.stem.replace("_history", "")
        histories[key] = h
    return histories


if __name__ == "__main__":
    compare_architectures()
