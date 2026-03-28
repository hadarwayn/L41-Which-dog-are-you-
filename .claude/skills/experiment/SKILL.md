# Experiment Runner

Run one of the 3 project experiments.

## Usage
```
/experiment compare      # Experiment 1: Architecture comparison
/experiment animals      # Experiment 2: Non-dog image predictions
/experiment celebrities  # Experiment 3: Celebrity-dog matching
```

## Experiments

### 1. Architecture Comparison
- Aggregates metrics from all trained models
- Generates comparison table (accuracy, params, time)
- Creates overlaid accuracy/loss plots
- Identifies best model

### 2. Non-Dog Predictions
- Loads best trained model
- Runs inference on animal images (`data/processed/test/animals/`)
- Records predicted breed + confidence per image
- Analyzes confidence distribution

### 3. Celebrity-Dog Matching
- Loads best trained model
- Runs inference on celebrity images (`data/processed/test/humans/`)
- Records top-3 predicted breeds per person
- Creates side-by-side gallery

## Commands
```bash
python -m src.experiments.compare_architectures
python -m src.experiments.non_dog_predictions
python -m src.experiments.celebrity_matching
```
