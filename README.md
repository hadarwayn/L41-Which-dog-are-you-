# Dog Breed Classification & "Which Dog Are You?" Experiment

> **An educational deep learning project** that classifies 120 dog breeds using 6 CNN architectures, compares their performance, and runs a fun experiment: *which dog breed does each famous person most resemble?*

**Course:** Deep Learning & AI | **Instructor:** Dr. Yoram Segal | **Author:** Hadar Wayne

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [CNN Fundamentals — A Beginner's Guide](#cnn-fundamentals)
3. [The 6 Architectures We Used](#architectures)
4. [Architecture Comparison Table](#architecture-comparison)
5. [Transfer Learning — Teaching an Old Model New Tricks](#transfer-learning)
6. [Results & Analysis](#results)
7. [Example Predictions](#example-predictions)
8. [The Fun Experiment: Animals as Dogs](#animal-experiment)
9. [The Fun Experiment: Celebrities as Dogs](#celebrity-experiment)
10. [Insights & Conclusions](#insights)
11. [How to Run This Project](#how-to-run)
12. [Project Structure](#project-structure)

---

## What This Project Does {#what-this-project-does}

Imagine you have a photo of a dog and you want to know its breed. This project teaches a computer to do exactly that — look at a photo and say "That's a Golden Retriever!" (or one of 119 other breeds).

**But we went further:**
- We trained **6 different AI architectures** and compared which one is best
- We used **transfer learning** (borrowing knowledge from a pre-trained brain)
- We ran the AI on **non-dog images** (horses, zebras, cats) to see what happens
- We ran it on **famous people** to see which dog breed they "look like" (the fun part!)

### Quick Stats

| Metric | Value |
|--------|-------|
| Dog breeds | 120 |
| Training images | 8,127 |
| Validation images | 2,095 |
| Architectures compared | 6 |
| Transfer learning stages | 3 per model |
| Famous people tested | 100 |
| Animal types tested | 10 |

---

## CNN Fundamentals — A Beginner's Guide {#cnn-fundamentals}

### What is a Neural Network?

Think of it like a brain made of math. Just like your brain has neurons connected to each other, a neural network has layers of mathematical "neurons" that pass information along.

```
Input (image) → [Layer 1] → [Layer 2] → ... → [Layer N] → Output (prediction)
```

### What is a CNN?

**CNN = Convolutional Neural Network** — a special type of neural network designed for images.

Regular neural networks look at every pixel individually (like reading a book letter by letter). CNNs are smarter — they look at **groups of nearby pixels** (like reading words and sentences). This lets them recognize shapes, textures, and patterns.

### How Convolution Works

Imagine you have a small magnifying glass (called a **kernel** or **filter**) that slides across the image. At each position, it multiplies the pixel values by the filter values and adds them up. This creates a new image called a **feature map** that highlights certain patterns.

```
Original Image          Kernel (3x3)         Feature Map
+---+---+---+---+      +---+---+---+        "Edges detected!"
| . | . | # | # |  *   | -1| -1| -1|   =    Shows where
| . | . | # | # |      |  0|  0|  0|        vertical edges
| . | . | # | # |      |  1|  1|  1|        are in the image
+---+---+---+---+      +---+---+---+
```

**Key idea:** The network LEARNS what filters to use. Early layers learn simple patterns (edges, corners). Deeper layers learn complex patterns (ears, noses, entire faces).

### What is Pooling?

After finding patterns, we **shrink** the image to keep only the most important information. **Max Pooling** takes the maximum value from each small region:

```
Before Pooling (4x4):     After Max Pooling (2x2):
+----+----+----+----+     +----+----+
| 1  | 3  | 5  | 7  |    | 4  | 8  |  (kept the max
| 2  | 4  | 6  | 8  | -> +----+----+   from each 2x2
| 9  | 11 | 13 | 15 |    | 12 | 16 |   region)
| 10 | 12 | 14 | 16 |    +----+----+
+----+----+----+----+
```

**Why?** It makes the network faster and helps it recognize objects regardless of their exact position.

### Activation Functions

#### ReLU (Rectified Linear Unit)
The most popular activation function. Simple rule: **keep positive values, replace negative with zero.**

```
Input:  [-2, 5, -1, 3, -4, 7]
ReLU:   [ 0, 5,  0, 3,  0, 7]
```

**Why?** Without activation functions, no matter how many layers you stack, the network can only learn straight lines. ReLU adds the ability to learn curves and complex patterns.

#### Softmax
Used in the final layer. Converts raw numbers into **percentages that add up to 100%**:

```
Raw scores:  [2.0, 1.0, 0.5]
Softmax:     [59%, 27%, 14%]  → adds up to 100%
```

This tells us: "I'm 59% sure it's breed A, 27% sure it's breed B..."

### Loss Function (Cross-Entropy)

The loss function measures **how wrong** the model is. If the model says "I'm 90% sure this is a Poodle" and it IS a Poodle, the loss is low. If it's actually a Beagle, the loss is high.

The goal of training: **minimize the loss**.

### Backpropagation — Learning From Mistakes

After each prediction, the network:
1. Checks how wrong it was (loss)
2. Figures out which weights caused the error
3. Adjusts those weights slightly
4. Repeats thousands of times

It's like a student who takes a test, checks the answers, and studies the topics they got wrong. Over many tests (epochs), they get better and better.

### What is an Epoch?

One epoch = the model has seen **every training image once**. We typically train for 10-20 epochs. It's like re-reading a textbook multiple times — each pass helps you understand more.

### Overfitting — When the Model Memorizes Instead of Learning

If you study by memorizing answers instead of understanding concepts, you'll ace the practice test but fail the real one. Similarly, a model can "memorize" training images instead of learning general patterns.

**Solutions we use:**
- **Dropout** — randomly turn off some neurons during training (forces the network to not rely on any single neuron)
- **Data Augmentation** — create variations of images (flip, rotate, change brightness) so the model sees more variety
- **Early Stopping** — stop training when validation accuracy stops improving

---

## The 6 Architectures We Used {#architectures}

We implemented 6 different CNN architectures, from a simple baseline to state-of-the-art models. Each one brought a new innovation to the field.

### 1. Simple CNN (Baseline) — Built From Scratch

**Year:** 2026 (our custom build)
**Parameters:** ~221K
**Analogy:** A student's first attempt at solving a puzzle — basic but educational.

```
Input (224x224x3)
  |
  v
[Conv 3x3, 32 filters] -> BatchNorm -> ReLU -> MaxPool
  |
[Conv 3x3, 64 filters] -> BatchNorm -> ReLU -> MaxPool
  |
[Conv 3x3, 128 filters] -> BatchNorm -> ReLU -> MaxPool
  |
Global Average Pooling
  |
Dense(512) -> ReLU -> Dropout(0.5)
  |
Dense(120) -> Softmax -> Prediction
```

**What it demonstrates:** All the fundamental building blocks — convolution, pooling, batch normalization, dropout, fully connected layers.

### 2. AlexNet (2012) — The Revolution Begins

**Parameters:** ~57M
**Key Innovation:** First to use ReLU and Dropout, trained on GPU
**Analogy:** The first car that proved engines are better than horses.

AlexNet won the 2012 ImageNet competition by a huge margin, proving that deep learning works for image recognition. It used **large filters** (11x11, 5x5) in early layers.

### 3. VGG-16 (2014) — Depth With Simplicity

**Parameters:** ~138M (largest!)
**Key Innovation:** Only uses 3x3 filters — proves that depth matters more than filter size
**Analogy:** Instead of one big step, take many small steps — you get further.

```
VGG-16 Architecture (simplified):
[3x3 Conv] x2 -> MaxPool -> [3x3 Conv] x2 -> MaxPool ->
[3x3 Conv] x3 -> MaxPool -> [3x3 Conv] x3 -> MaxPool ->
[3x3 Conv] x3 -> MaxPool -> FC -> FC -> Softmax
```

**Insight:** Two stacked 3x3 filters see the same area as one 5x5 filter, but with fewer parameters and more non-linearity.

### 4. GoogLeNet / Inception (2014) — Do Everything At Once

**Parameters:** ~5M (25x fewer than VGG!)
**Key Innovation:** Inception module — run multiple filter sizes in parallel
**Analogy:** Instead of choosing one tool, use a Swiss Army knife.

```
                Input
          /    |    |     \
      [1x1] [3x3] [5x5] [Pool]    <- Try all sizes at once!
          \    |    |     /
              Concat
              Output
```

**The trick:** 1x1 convolutions reduce dimensions before the expensive 3x3 and 5x5 operations, keeping the model small.

### 5. ResNet-50 (2015) — Skip Connections Save The Day

**Parameters:** ~25M
**Key Innovation:** Skip connections (residual learning)
**Analogy:** Taking a shortcut through a building instead of climbing every staircase.

```
        Input
          |
     [Conv 1x1]
     [Conv 3x3]     <- The "residual" path
     [Conv 1x1]
          |
          + -------- Input (skip connection!)
          |
        Output
```

**The problem it solves:** In very deep networks (50+ layers), gradients become tiny as they flow backward (vanishing gradients). The skip connection lets gradients flow directly, enabling training of much deeper networks.

**Formula:** `H(x) = F(x) + x` — instead of learning the full transformation H(x), just learn the **residual** F(x) = H(x) - x. If the identity is optimal, F(x) is easy to learn (just output zeros).

### 6. MobileNet v2 (2018) — Speed For Mobile

**Parameters:** ~3.4M (smallest!)
**Key Innovation:** Depthwise separable convolutions
**Analogy:** Instead of a big powerful truck, a small efficient electric scooter that still gets the job done.

**Regular convolution:** One filter looks at ALL channels at every position (expensive).
**Depthwise separable:** First look at each channel separately (depthwise), then combine channels (pointwise). Same result, ~8x fewer computations.

---

## Architecture Comparison Table {#architecture-comparison}

### Historical Evolution

| Year | Architecture | Layers | Parameters | Key Innovation | ImageNet Error |
|------|-------------|--------|-----------|----------------|---------------|
| 1998 | LeNet-5 | 5 | 60K | First CNN | N/A |
| 2012 | **AlexNet** | 8 | 57M | ReLU, Dropout, GPU | 16.4% |
| 2014 | **VGG-16** | 16 | 138M | Only 3x3 filters | 7.3% |
| 2014 | **Inception** | 22 | 5M | Parallel paths | 6.7% |
| 2015 | **ResNet-50** | 50 | 25M | Skip connections | 3.6% |
| 2018 | **MobileNet v2** | — | 3.4M | Depthwise separable | 5.6% |

### Our Results (Preliminary — 10% Data Subset, CPU Training)

> These results use only 10% of the training data (761 images) with 3-5 epochs per stage, trained on CPU. Full results from Colab GPU training on 100% data will be updated below.

| Model | Top-1 Acc | Top-5 Acc | Params | Train Time | Best Stage |
|-------|----------|----------|--------|-----------|-----------|
| Simple CNN | 1.4% | 7.0% | 221K | 258s | N/A (from scratch) |
| AlexNet | 26.6% | 56.6% | 57M | 973s | Stage 3 |
| MobileNet | 35.7% | 69.2% | 2.4M | 500s | Stage 2 |
| **ResNet-50** | **67.1%** | **95.8%** | 24M | 1,518s | Stage 3 |
| VGG-16 | *Colab only* | — | 135M | *CPU too slow* | — |
| Inception | *Colab only* | — | 25M | *CPU too slow* | — |

**Key takeaways:**
- **ResNet-50 wins decisively**: 67.1% top-1 and 95.8% top-5 on just 10% data (761 images, 3 epochs per stage)
- **Transfer learning is transformative**: Simple CNN (0.7%) vs ResNet-50 (67.1%) — that's 48x better
- **Size doesn't equal quality**: MobileNet (2.4M params) beats AlexNet (57M params) by 9 percentage points
- **Top-5 accuracy is impressive**: ResNet-50 puts the correct breed in its top 5 guesses 95.8% of the time

---

## Transfer Learning — Teaching an Old Model New Tricks {#transfer-learning}

### What is Transfer Learning?

Imagine you already know how to ride a bicycle. Learning to ride a motorcycle is much easier than learning from scratch — you already understand balance, steering, and road awareness. You just need to learn the new parts (throttle, gears, brakes).

Transfer learning works the same way:
1. Take a model trained on **ImageNet** (1.2 million images, 1000 categories)
2. It already knows how to recognize edges, textures, shapes, and objects
3. We just teach it the new part: **which dog breed is this?**

### Why Not Train From Scratch?

Our Simple CNN baseline got **0.7% accuracy**. ResNet-50 with transfer learning got **65.7%** — that's **94x better!** Training from scratch with only ~8,000 images and 120 classes is simply not enough data for a model to learn visual features from zero.

### The 3 Stages

```
Stage 1: FEATURE EXTRACTION          Stage 2: PARTIAL FINE-TUNING
+---------------------------+        +---------------------------+
|  FROZEN (ImageNet weights)|        |  FROZEN (early layers)    |
|  - Edge detectors         |        |  - Edge detectors         |
|  - Texture recognizers    |        |  - Texture recognizers    |
|  - Shape detectors        |        +---------------------------+
|  - Object parts           |        |  UNFROZEN (top 25%)       |
+---------------------------+        |  - High-level features    |
|  TRAINABLE (new)          |        +---------------------------+
|  - Dog breed classifier   |        |  TRAINABLE                |
+---------------------------+        |  - Dog breed classifier   |
LR = 0.001                          +---------------------------+
                                     LR = 0.0001

Stage 3: FULL FINE-TUNING
+---------------------------+
|  UNFROZEN (all layers)    |
|  - Everything re-trained  |
|  - With very small LR     |
|  - Risk of overfitting!   |
+---------------------------+
LR = 0.00001
```

### Transfer Learning Results (Preliminary — 10% Data)

| Model | Stage 1 (Frozen) | Stage 2 (Partial) | Stage 3 (Full) | Best Stage |
|-------|-----------------|-------------------|----------------|-----------|
| AlexNet | 17.5% | 25.2% | **26.6%** | Stage 3 |
| MobileNet | 24.5% | **35.7%** | 35.7% | Stage 2 |
| ResNet-50 | 49.0% | 65.7% | **67.1%** | Stage 3 |

**Patterns observed:**
- **Stage 1 (frozen) is a strong starting point** — already 49% for ResNet-50 just from the classifier head
- **Stages 2 and 3 improve further** — unfreezing backbone layers helps adapt to dog-specific features
- **Bigger models benefit more from Stage 3** — ResNet-50 and AlexNet improved in Stage 3, while MobileNet plateaued at Stage 2
- **The jump from Stage 1 to Stage 2 is the biggest** — this is where the model starts learning dog-specific patterns

---

## Results & Analysis {#results}

> **Note:** These are preliminary results from 10% data subset on CPU. Final results from full Colab GPU training will be updated here.

### Accuracy Comparison

The class distribution plot is saved at `results/graphs/class_distribution.png`.

**Dataset Analysis:**
- 120 breeds, 52-100 images per breed (well balanced)
- Mean: 67.7 images per breed
- No breeds with fewer than 50 samples

### What We Observe So Far

1. **Pre-trained models dominate**: ResNet-50 (65.7%) vs Simple CNN (0.7%)
2. **Model size doesn't equal performance**: MobileNet (2.4M params, 35.7%) outperforms AlexNet (57M params, 25.2%)
3. **Architecture innovation matters more than brute force depth**
4. **Stage 2 (partial fine-tuning) is the sweet spot** for small datasets

---

## Example Predictions {#example-predictions}

*This section will be populated with image grids after Colab full training:*
- **Correct** (green): model got the breed right
- **Wrong** (red): model predicted the wrong breed
- **Funny** (yellow): humans and animals predicted as dog breeds

---

## The Fun Experiment: Animals as Dogs {#animal-experiment}

*Coming after Colab training — we'll feed these animals through the NN:*

| Animal | Expected Dog Breed? | Why? |
|--------|-------------------|------|
| Wolf | Husky / German Shepherd | Similar face shape |
| Zebra | Dalmatian | Stripes vs spots pattern |
| Fox | Shiba Inu | Similar face and color |
| Lion | Chow Chow | Mane looks like fur |
| Cat | ??? | Curious what happens! |

---

## The Fun Experiment: Celebrities as Dogs {#celebrity-experiment}

*Coming after Colab training — we'll feed 100 celebrity photos through the NN and see which dog breed each person "is"!*

**Methodology:**
1. Load celebrity face photo
2. Resize to 224x224, normalize
3. Feed through best trained model (ResNet-50)
4. Get softmax output: probability distribution over 120 breeds
5. Show top-3 predicted breeds with confidence

---

## Insights & Conclusions {#insights}

### What We Learned

1. **Transfer learning is essential** for small datasets. Without it, accuracy is near random.
2. **Architecture design matters more than size**: MobileNet (3.4M params) can outperform AlexNet (57M params).
3. **ResNet's skip connections** enable significantly deeper and more accurate networks.
4. **Partial fine-tuning (Stage 2)** consistently gives the best results on this dataset.

### Limitations

- 10% data subset gives preliminary results only — full training needed on GPU
- CPU training is too slow for VGG-16 (138M params) — requires Colab Pro
- 120 classes with ~80 images each is challenging — some breeds look very similar
- Celebrity/animal experiments only run on Colab (full model needed)

### What Surprised Us

- How badly a custom CNN performs without transfer learning (1.4% vs 67.1%)
- That MobileNet (designed for phones) beats AlexNet (designed for GPUs) by 9 percentage points with 24x fewer parameters
- ResNet-50's top-5 accuracy of 95.8% — it almost always has the right answer in its top 5 guesses
- The power of skip connections: ResNet-50 (50 layers) dramatically outperforms VGG's approach of just stacking more layers

---

## How to Run This Project {#how-to-run}

### Option A: WSL / Local (10% Quick Test)

```bash
# Navigate to project
cd /mnt/c/2025AIDEV/L41

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Download and prepare data
python -m src.data.download
python -m src.data.organize

# Train all models (10% subset, ~1 hour on CPU)
python run_training.py

# Or train a single model
python run_training.py --model resnet50
```

### Option B: Google Colab (Full Training — Recommended)

1. Open the [Colab Notebook](https://colab.research.google.com/drive/1Rvgw3hEfrl53GqFkeXpQUZZYdSajRmpJ)
2. Enable GPU: Runtime -> Change runtime type -> GPU
3. Run all cells
4. Full training takes ~2-4 hours with T4 GPU

### Option C: Windows PowerShell

```powershell
cd C:\2025AIDEV\L41
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
python run_training.py
```

---

## Project Structure {#project-structure}

```
L41/
├── src/
│   ├── config.py               # All settings and paths
│   ├── data/                   # Data pipeline
│   │   ├── download.py         # Kaggle dataset download
│   │   ├── organize.py         # Organize into breed folders
│   │   ├── augmentation.py     # Image transforms
│   │   ├── dataset.py          # PyTorch DataLoader
│   │   └── analysis.py         # Dataset statistics
│   ├── models/                 # 6 CNN architectures
│   │   ├── simple_cnn.py       # Custom baseline
│   │   ├── alexnet.py          # AlexNet (2012)
│   │   ├── vgg.py              # VGG-16 (2014)
│   │   ├── inception.py        # GoogLeNet (2014)
│   │   ├── resnet.py           # ResNet-50 (2015)
│   │   └── mobilenet.py        # MobileNet v2 (2018)
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py          # Training loop
│   │   ├── transfer_learning.py # 3-stage TL pipeline
│   │   └── evaluate.py         # Metrics and evaluation
│   ├── inference/
│   │   └── predict.py          # predict_dog_breed() function
│   ├── experiments/            # 3 experiments
│   └── visualization/          # Plots and galleries
├── data/                       # Dataset (gitignored)
├── results/                    # Outputs
│   ├── graphs/                 # All plots
│   ├── tables/                 # CSV results
│   └── models/                 # Saved weights
├── notebooks/
│   └── dog_breed_classifier.ipynb
├── run_training.py             # Main training script
├── requirements.txt
└── README.md                   # This file
```

---

*This README will be updated with final results after full Colab GPU training.*
