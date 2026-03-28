# Visualization Generator

Generate project visualizations and result outputs.

## Usage
```
/visualize plots          # Accuracy/loss comparison plots
/visualize confusion      # Confusion matrix for best model
/visualize gallery        # Celebrity-dog match gallery
/visualize all            # Generate everything
```

## What Gets Generated

### Plots (`results/graphs/`)
- `accuracy_comparison.png` — All models' accuracy vs epochs
- `loss_comparison.png` — All models' loss vs epochs
- `transfer_learning_stages.png` — Stage 1/2/3 bar chart per model
- `class_distribution.png` — Samples per breed histogram
- `sample_predictions.png` — Correct/incorrect prediction grid

### Tables (`results/tables/`)
- `model_comparison.csv` — Model | Acc | Params | Time
- `transfer_learning_comparison.csv` — Model | Stage1 | Stage2 | Stage3
- `non_dog_predictions.csv` — Image | Predicted Breed | Confidence
- `celebrity_dog_matches.csv` — Celebrity | Top-3 Breeds | Confidence

### Gallery (`results/experiments/celebrity_gallery/`)
- Side-by-side celebrity → dog breed match images

## Commands
```bash
python -m src.visualization.plots
python -m src.visualization.confusion
python -m src.visualization.gallery
```
