# ML Debugger & Root Cause Analysis

Debug training failures, data pipeline issues, and unexpected model behavior.

## Usage
```
/debug <description_of_issue>
/debug <file_or_module> [--data] [--training] [--model]
```

## The Debugging Mindset

Debugging is **investigation, not guessing**. Every action tests a specific hypothesis.

### The Loop: Observe -> Hypothesize -> Act -> Observe

1. **Observe** — Error messages, loss curves, accuracy numbers, data samples
2. **Hypothesize** — "I believe X fails because Y"
3. **Act** — Add logging, check tensor shapes, visualize data, run subset
4. **Observe** — Confirm or disprove, refine and repeat

## Common ML Debug Paths

### Training Loss Not Decreasing
```
1. Check learning rate (too high? too low?)
2. Verify labels match images (visualize batch with labels)
3. Check loss function (CrossEntropy needs class indices, not one-hot)
4. Verify model output shape matches num_classes
5. Check if model is in train mode: model.train()
6. Verify optimizer.zero_grad() is called each step
```

### GPU Out of Memory
```
1. Reduce batch_size
2. Use torch.cuda.empty_cache() between models
3. Enable mixed precision (autocast + GradScaler)
4. Use gradient accumulation for effective larger batch
5. For VGG-16: batch_size=16 may be needed
```

### Data Pipeline Issues
```
1. Visualize sample batch: show images with labels
2. Check tensor shapes: print(batch.shape, labels.shape)
3. Verify normalization: values should be ~[-2, 2] after ImageNet norm
4. Check class distribution: are weights computed correctly?
5. Verify no data leakage between train/val
```

### Transfer Learning Not Improving
```
1. Verify layers are actually frozen: check requires_grad
2. Check learning rate per stage (too high unfreezes chaos)
3. Ensure classifier head matches num_classes (120, not 1000)
4. Verify pre-trained weights loaded (not random init)
```

### Model Predicts Same Class for Everything
```
1. Class imbalance — model learned majority class
2. Learning rate too high — weights diverged
3. Wrong loss function or missing softmax
4. Labels are wrong (verify data pipeline)
```
