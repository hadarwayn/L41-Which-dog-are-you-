# Model Training Skill

Train a specific model architecture with transfer learning.

## Usage
```
/train <model_name> [--data-fraction 0.1] [--epochs 5] [--stages 3]
```

## Examples
```
/train resnet50 --data-fraction 0.1 --epochs 3
/train vgg16 --stages 1
/train all --data-fraction 0.1 --epochs 3
```

## What This Skill Does

1. **Loads model** from `src/models/` factory
2. **Prepares data** with correct transforms and DataLoader
3. **Runs transfer learning** through 3 stages:
   - Stage 1: Feature extraction (frozen backbone, lr=1e-3)
   - Stage 2: Partial fine-tuning (top 20-30% unfrozen, lr=1e-4)
   - Stage 3: Full fine-tuning (all layers, lr=1e-5)
4. **Logs metrics** per epoch (loss, accuracy, time)
5. **Saves best checkpoint** per stage
6. **Reports results** summary

## Available Models
- `simple_cnn` — Custom baseline (no transfer learning)
- `alexnet` — AlexNet with ImageNet weights
- `vgg16` — VGG-16 with ImageNet weights
- `inception` — Inception v3 (299x299 input)
- `resnet50` — ResNet-50 with ImageNet weights
- `mobilenet` — MobileNet v2 with ImageNet weights
- `all` — Train all models sequentially

## Commands
```bash
# WSL (10% data, quick test)
python -m src.training.trainer --model resnet50 --data-fraction 0.1 --epochs 3

# Evaluate after training
python -m src.training.evaluate --model resnet50
```
