# Test Runner

Run tests to validate code before training or deployment.

## Usage
```
/test data       # Test data pipeline
/test models     # Test all model architectures load correctly
/test training   # Smoke test: 1 epoch per model
/test all        # Run everything
```

## Test Categories

### Data Pipeline Tests
- Dataset download/extraction
- Folder organization and stratified split
- Resolution standardization (all images same size)
- Class weight computation
- DataLoader batch shapes
- Augmentation transforms

### Model Tests
- Each model loads with correct architecture
- Forward pass produces (batch, 120) output
- Pre-trained weights load without errors
- Classifier head replaced correctly
- Model save/load from checkpoint

### Training Tests
- Training loop runs 1 epoch without errors
- Loss decreases after 1 epoch (sanity)
- Metrics logged in correct JSON format
- Freeze/unfreeze works for transfer learning

## Commands
```bash
cd /mnt/c/2025AIDEV/L41
source .venv/bin/activate
python -m pytest tests/ -v
python -m pytest tests/test_data_pipeline.py -v
python -m pytest tests/test_models.py -v
python -m pytest tests/test_training.py -v --timeout=300
```
