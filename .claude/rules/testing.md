# Testing Standards

## Testing Philosophy

Every piece of code must be verifiable. For ML projects, "testing" includes:
1. **Unit tests** — individual functions work correctly
2. **Pipeline tests** — data flows end-to-end
3. **Smoke tests** — models train for 1 epoch without crashing
4. **Output validation** — results are in expected format/range

## What to Test

### Data Pipeline
- [ ] Dataset download and extraction works
- [ ] Images are organized into correct breed folders
- [ ] Train/val split is stratified (same breed proportions)
- [ ] All images resize to target resolution (224x224 or 299x299)
- [ ] Normalization uses correct ImageNet mean/std
- [ ] DataLoader produces correct batch shapes
- [ ] Class weights are computed correctly
- [ ] 10% subset maintains breed distribution

### Models
- [ ] Each model accepts correct input shape and outputs (batch, 120)
- [ ] Pre-trained weights load without errors
- [ ] Classifier head replacement works
- [ ] Forward pass doesn't crash on a sample batch
- [ ] Model can be saved and loaded from checkpoint

### Training
- [ ] Training loop runs for 1 epoch without errors
- [ ] Loss decreases after 1 epoch (sanity check)
- [ ] Metrics are logged correctly (JSON format)
- [ ] Checkpoint saves and loads correctly
- [ ] Transfer learning freeze/unfreeze works (check requires_grad)
- [ ] Early stopping triggers when validation loss plateaus

### Visualization
- [ ] Plots generate without errors
- [ ] Confusion matrix renders correctly
- [ ] Comparison tables have all expected columns
- [ ] Gallery images are readable

## Running Tests

```bash
# In WSL
cd /mnt/c/2025AIDEV/L41
source .venv/bin/activate

# Quick smoke test — all models
python -m pytest tests/ -v

# Test data pipeline
python -m pytest tests/test_data_pipeline.py -v

# Test models load correctly
python -m pytest tests/test_models.py -v

# Test training runs for 1 epoch
python -m pytest tests/test_training.py -v --timeout=300
```

## Smoke Test Before Colab

Before creating the Colab notebook, run this locally:
```bash
# Verify all models train on 10% data for 1 epoch each
python -m src.training.trainer --model simple_cnn --data-fraction 0.1 --epochs 1
python -m src.training.trainer --model alexnet --data-fraction 0.1 --epochs 1
python -m src.training.trainer --model vgg16 --data-fraction 0.1 --epochs 1
python -m src.training.trainer --model inception --data-fraction 0.1 --epochs 1
python -m src.training.trainer --model resnet50 --data-fraction 0.1 --epochs 1
python -m src.training.trainer --model mobilenet --data-fraction 0.1 --epochs 1
```
