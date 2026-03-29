# 🐕 Dog Breed Classification & "Which Dog Are You?" Experiment

> **An educational deep learning project** that classifies 120 dog breeds using 6 CNN architectures, compares their performance, and runs a fun experiment: *which dog breed does each famous person most resemble?*

**Course:** Deep Learning & AI | **Instructor:** Dr. Yoram Segal | **Author:** Hadar Wayne

---

## 📋 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [CNN Fundamentals](#-cnn-fundamentals--how-computers-see-images)
- [The 6 Architectures](#-the-6-architectures-we-compared)
- [Architecture Comparison](#-architecture-comparison)
- [Transfer Learning](#-transfer-learning--teaching-an-old-model-new-tricks)
- [Results & Analysis](#-results--analysis)
- [Animal Experiment](#-experiment-what-dog-breed-is-this-animal)
- [Celebrity Experiment](#-experiment-which-dog-breed-are-you)
- [Insights & Conclusions](#-insights--conclusions)
- [How to Run](#-how-to-run-this-project)
- [Project Structure](#-project-structure)

---

## 🎯 What This Project Does

Imagine you show a computer a photo and it tells you: *"That's a Golden Retriever!"* (or one of 119 other breeds). That's what we built.

**But we went way further:**

1. We trained **6 different AI architectures** and compared them
2. We used **transfer learning** (borrowing knowledge from pre-trained models)
3. We tested the AI on **animals that aren't dogs** (horses, zebras, cats...)
4. We tested it on **human faces** to see which dog breed they "resemble"

### Quick Stats

| Metric | Value |
|--------|-------|
| Dog breeds classified | **120** |
| Training images | **8,127** |
| Validation images | **2,095** |
| Architectures compared | **6** |
| Transfer learning stages | **3** per model |
| Best accuracy (ResNet-50) | **67.1%** (10% data) / **~85%+** (full, pending) |
| Top-5 accuracy (ResNet-50) | **95.8%** |

---

## 🧠 CNN Fundamentals — How Computers "See" Images

### What is a Neural Network?

Think of it like a **digital brain**. Just like your brain has billions of neurons connected to each other, a neural network has layers of mathematical "neurons" that pass information forward.

```
Photo of a dog  →  [Layer 1]  →  [Layer 2]  →  ...  →  [Final Layer]  →  "Golden Retriever!"
                    (edges)      (shapes)              (breed features)    (answer)
```

**Key idea:** Each layer learns something more complex than the previous one.

### What Makes a CNN Special?

**CNN = Convolutional Neural Network** — a type of neural network specifically designed for images.

A regular neural network looks at every pixel individually (like reading a book letter by letter). A CNN is smarter — it looks at **groups of nearby pixels** together (like reading words and sentences). This is why CNNs are so good with images.

### Convolution — The Core Operation

Imagine you have a small **magnifying glass** (called a **kernel** or **filter**, usually 3x3 pixels). You slide it across the entire image, one position at a time. At each position, you multiply the pixel values by the filter values and add them up. This creates a new image called a **feature map**.

```
  Original Image (5x5)        Filter (3x3)         Feature Map
  ┌───┬───┬───┬───┬───┐      ┌───┬───┬───┐        "I found
  │ 0 │ 0 │ 1 │ 1 │ 1 │      │-1 │-1 │-1 │         vertical
  ├───┼───┼───┼───┼───┤  ×   ├───┼───┼───┤   =     edges!"
  │ 0 │ 0 │ 1 │ 1 │ 1 │      │ 0 │ 0 │ 0 │
  ├───┼───┼───┼───┼───┤      ├───┼───┼───┤
  │ 0 │ 0 │ 1 │ 1 │ 1 │      │ 1 │ 1 │ 1 │
  └───┴───┴───┴───┴───┘      └───┴───┴───┘

  The filter slides across the image like a scanner,
  detecting patterns at every position.
```

**The magic:** The network **learns** which filters to use! Early layers learn to detect simple patterns (edges, corners). Deeper layers combine these into complex patterns (eyes, ears, fur textures).

### Pooling — Shrinking While Keeping the Important Stuff

After detecting patterns, we **shrink** the image to keep only the strongest signals. **Max Pooling** takes the maximum value from each small region:

```
  Before (4x4)              After Max Pooling (2x2)
  ┌────┬────┬────┬────┐     ┌────┬────┐
  │  1 │  3 │  5 │  7 │     │  4 │  8 │   ← Kept the biggest
  │  2 │  4 │  6 │  8 │ →   ├────┼────┤     value from each
  │  9 │ 11 │ 13 │ 15 │     │ 12 │ 16 │     2×2 region
  │ 10 │ 12 │ 14 │ 16 │     └────┴────┘
  └────┴────┴────┴────┘
```

**Why?** Makes the network faster and helps it recognize objects regardless of their exact position in the image.

### ReLU — The On/Off Switch

**ReLU (Rectified Linear Unit)** is the most common activation function. Simple rule: **keep positive values, replace negatives with zero.**

```
  Input:   [-2,  5, -1,  3, -4,  7]
  After ReLU: [ 0,  5,  0,  3,  0,  7]
```

**Why it matters:** Without activation functions, stacking layers is pointless — mathematically it collapses to a single layer. ReLU introduces non-linearity, letting the network learn complex curved decision boundaries.

### Softmax — Turning Numbers into Percentages

The final layer outputs raw scores for each breed. **Softmax** converts these into probabilities that add up to 100%:

```
  Raw scores:    [2.0,   1.0,   0.5,   0.1,  ...]    (120 numbers)
                   ↓      ↓      ↓      ↓
  After Softmax: [59%,   27%,   10%,    4%,  ...]    (adds to 100%)
                   ↓
  Prediction: "59% sure it's a Golden Retriever"
```

### Loss Function — How Wrong Are We?

The **Cross-Entropy Loss** measures how far the model's prediction is from the correct answer:
- Model says 90% Golden Retriever and it IS a Golden Retriever → **low loss** (good!)
- Model says 10% Golden Retriever and it IS a Golden Retriever → **high loss** (bad!)

**Goal of training: minimize the loss.**

### Backpropagation — Learning From Mistakes

After each prediction, the network:
1. **Checks** how wrong it was (calculates loss)
2. **Traces back** through every layer to find which weights caused the error
3. **Adjusts** those weights slightly in the right direction
4. **Repeats** thousands of times

```
  Forward pass:  Image → Layers → Prediction → Loss
                                                  ↓
  Backward pass: Adjust weights ← Gradients ← Loss
                 (learning!)      (blame assignment)
```

It's like a student who takes a test, checks the answers, and studies the topics they got wrong. Over many tests (epochs), they improve.

### What is an Epoch?

One **epoch** = the model has seen every training image once. We train for 10-15 epochs, like re-reading a textbook multiple times.

### Overfitting — Memorizing vs Understanding

If you study by memorizing answers word-for-word, you'll ace the practice test but fail the real exam. Neural networks can do the same thing — **memorize** the training images instead of learning general patterns.

```
  Training accuracy keeps going up  📈
  Validation accuracy stops or drops 📉  ← OVERFITTING!
```

**Solutions we use:**
- **Dropout** — randomly turn off 50% of neurons during training (forces the network to not rely on any single neuron)
- **Data Augmentation** — flip, rotate, zoom, change brightness of images (creates variety)
- **Early Stopping** — stop training when validation accuracy stops improving

### Data Augmentation — Creating Variety

One image of a dog becomes many:

```
  Original    →  Flipped    →  Rotated    →  Brighter   →  Zoomed
  🐕           🐕(mirror)    🐕(tilted)    🐕(light)     🐕(close-up)
```

This helps the model learn that a dog is a dog regardless of angle, lighting, or position.

---

## 🏗️ The 6 Architectures We Compared

We trained 6 different CNN architectures, each representing a milestone in deep learning history.

### Architecture Timeline

```
  1998          2012          2014          2014          2015          2018
   │             │             │             │             │             │
   ▼             ▼             ▼             ▼             ▼             ▼
  LeNet      AlexNet        VGG-16     GoogLeNet     ResNet-50     MobileNet
  (first)    (GPU+ReLU)    (3×3 only)  (parallel)   (skip conn.)  (mobile)
  60K params  57M params   138M params   5M params    25M params   3.4M params
```

---

### 1. Simple CNN (Our Baseline)

**What:** A basic CNN we built from scratch — no pre-trained knowledge.
**Analogy:** A student taking an exam on day one of school, with zero preparation.

```
  Input Image (224×224×3)
       │
       ▼
  ┌─────────────────────┐
  │ Conv 3×3 (32 filters)│ → BatchNorm → ReLU → MaxPool
  └─────────┬───────────┘
            ▼
  ┌─────────────────────┐
  │ Conv 3×3 (64 filters)│ → BatchNorm → ReLU → MaxPool
  └─────────┬───────────┘
            ▼
  ┌─────────────────────┐
  │ Conv 3×3(128 filters)│ → BatchNorm → ReLU → MaxPool
  └─────────┬───────────┘
            ▼
  Global Average Pooling → Dense(512) → Dropout → Dense(120) → Softmax
                                                        │
                                                   "golden_retriever"
```

**Parameters:** 221,304 | **Our accuracy:** 1.4% (basically random guessing with 120 breeds!)

---

### 2. AlexNet (2012) — The Deep Learning Revolution

**What:** The architecture that proved deep learning works. Won ImageNet 2012 by a huge margin.
**Key Innovation:** First to use ReLU (instead of slow sigmoid) and Dropout for regularization.
**Analogy:** The first car that proved engines are faster than horses.

**Parameters:** 57.5M | **Our accuracy:** 26.6% (Stage 3) | **Top-5:** 56.6%

---

### 3. VGG-16 (2014) — Deep and Simple

**What:** 16 layers using ONLY 3×3 filters. Proved that going deeper with small filters beats using large filters.
**Key Innovation:** Uniform architecture — just stack 3×3 convolutions and max pooling.
**Analogy:** Instead of climbing a wall in one big jump, take many small steps on a ladder.

```
  VGG-16:
  [3×3 Conv]×2 → Pool → [3×3 Conv]×2 → Pool → [3×3 Conv]×3 → Pool →
  [3×3 Conv]×3 → Pool → [3×3 Conv]×3 → Pool → FC → FC → Softmax
```

**Insight:** Two stacked 3×3 filters see the same area as one 5×5 filter, but with fewer parameters and more non-linearity.

**Parameters:** 138M (largest!) | **Our accuracy:** 69.6% (Colab, Stage 3)

---

### 4. GoogLeNet / Inception (2014) — Do Everything At Once

**What:** Instead of choosing one filter size, use ALL sizes in parallel!
**Key Innovation:** The Inception module — parallel 1×1, 3×3, 5×5 convolutions + pooling.
**Analogy:** Instead of choosing one tool, use a Swiss Army knife.

```
                     Input
              ┌────┬───┬───┬────┐
              │    │   │   │    │
            [1×1][1×1][1×1][Pool]
              │    │   │   │
              │  [3×3][5×5] │
              │    │   │   │
              └────┴───┴───┘
                  Concatenate
                     │
                   Output
```

**The trick:** 1×1 convolutions reduce dimensions first (bottleneck), making 3×3 and 5×5 operations much cheaper.

**Parameters:** 25M (5× fewer than VGG!) | **Our accuracy:** Pending (Colab running)

---

### 5. ResNet-50 (2015) — Skip Connections Solve Everything

**What:** 50 layers with "shortcut" connections that skip over layers.
**Key Innovation:** Residual learning — instead of learning the full transformation, learn only the **difference** (residual).
**Analogy:** Taking an elevator shortcut instead of climbing every staircase in a 50-story building.

```
        Input (x)
          │
     ┌────┴────┐
     │         │
  [Conv 1×1]   │
  [Conv 3×3]   │  ← The "residual" path (learns the difference)
  [Conv 1×1]   │
     │         │
     └────┬────┘
          │
       x + F(x)     ← Skip connection adds input directly!
          │
        Output

  Formula: H(x) = F(x) + x

  Instead of learning H(x) from scratch,
  just learn F(x) = H(x) - x (the residual).
  If nothing needs to change, F(x) = 0 is easy to learn.
```

**Why it works:** In very deep networks, gradients become tiny as they flow backward (**vanishing gradients**). The skip connection lets gradients flow directly through, enabling 50+ layer networks.

**Parameters:** 25M | **Our accuracy:** 67.1% (10% data) | **Top-5:** 95.8%

---

### 6. MobileNet v2 (2018) — Speed for Phones

**What:** Designed for mobile devices — fast inference with minimal parameters.
**Key Innovation:** Depthwise separable convolutions — split convolution into two cheaper operations.
**Analogy:** Instead of a powerful but heavy truck, a lightweight electric scooter that still gets the job done.

**Regular convolution:** One filter processes ALL channels at every position (expensive).
**Depthwise separable:** First process each channel separately (depthwise), then combine (pointwise). Same result, ~8× fewer computations!

**Parameters:** 2.4M (smallest!) | **Our accuracy:** 35.7% (10% data) | **Top-5:** 69.2%

---

## 📊 Architecture Comparison

### Historical Overview

| Year | Architecture | Layers | Parameters | Key Innovation | ImageNet Error |
|------|-------------|--------|-----------|----------------|---------------|
| 1998 | LeNet-5 | 5 | 60K | First practical CNN | — |
| 2012 | **AlexNet** | 8 | 57M | ReLU, Dropout, GPU | 16.4% |
| 2014 | **VGG-16** | 16 | 138M | Only 3×3 filters | 7.3% |
| 2014 | **Inception** | 22 | 5M | Parallel paths | 6.7% |
| 2015 | **ResNet-50** | 50 | 25M | Skip connections | 3.6% |
| 2018 | **MobileNet** | — | 3.4M | Depthwise separable | 5.6% |

### Our Results

> **Preliminary results** from 10% data subset (761 training images, 3 epochs per stage, CPU). Full results from Colab training on 100% data will be updated below.

| Model | Top-1 Accuracy | Top-5 Accuracy | Parameters | Training Time |
|-------|:-------------:|:--------------:|:----------:|:------------:|
| Simple CNN | 1.4% | 7.0% | 221K | 4 min |
| AlexNet | 26.6% | 56.6% | 57M | 16 min |
| **MobileNet** | **35.7%** | **69.2%** | **2.4M** | 8 min |
| **ResNet-50** | **67.1%** | **95.8%** | **25M** | 25 min |

> VGG-16 and Inception results from full Colab training (100% data):

| Model | Top-1 Accuracy | Status |
|-------|:-------------:|:------:|
| VGG-16 | 69.6% | Done (Colab) |
| Inception | Training... | In progress |

### Accuracy Comparison Chart

![Accuracy Comparison](results/graphs/accuracy_comparison.png)

### Architecture Comparison Chart

![Architecture Comparison](results/graphs/architecture_comparison.png)

### Class Distribution

The dataset is well-balanced: 52-100 images per breed, mean of 67.7.

![Class Distribution](results/graphs/class_distribution.png)

---

## 🔄 Transfer Learning — Teaching an Old Model New Tricks

### What is Transfer Learning?

Imagine you already know how to **ride a bicycle**. Learning to ride a **motorcycle** is much easier than learning from zero — you already understand balance, steering, and road awareness. You just need to learn the throttle and gears.

Transfer learning works the same way:
1. Start with a model pre-trained on **ImageNet** (1.2 million images, 1000 categories)
2. It already knows edges, textures, shapes, and objects
3. Just teach it the new part: **which dog breed is this?**

### Why Not Train From Scratch?

| Approach | Accuracy | Difference |
|----------|:--------:|:----------:|
| Simple CNN (from scratch) | 1.4% | Baseline |
| ResNet-50 (transfer learning) | 67.1% | **48× better!** |

With only ~8,000 training images and 120 classes, training from scratch simply doesn't have enough data to learn visual features from zero.

### The 3 Stages

```
  STAGE 1: Feature Extraction       STAGE 2: Partial Fine-Tuning    STAGE 3: Full Fine-Tuning
  ┌─────────────────────┐           ┌─────────────────────┐         ┌─────────────────────┐
  │    FROZEN ❄️         │           │    FROZEN ❄️         │         │    UNFROZEN 🔥       │
  │  (ImageNet weights)  │           │  (early layers)      │         │  (all layers)        │
  │  Edge detectors      │           │  Edge detectors      │         │  Everything re-tuned │
  │  Texture recognizers │           ├─────────────────────┤         │  with very small LR  │
  │  Shape detectors     │           │    UNFROZEN 🔥       │         │                      │
  ├─────────────────────┤           │  (top 25% layers)    │         │  Risk: overfitting!  │
  │    TRAINABLE 🔥      │           ├─────────────────────┤         ├─────────────────────┤
  │  New breed classifier│           │    TRAINABLE 🔥      │         │    TRAINABLE 🔥      │
  └─────────────────────┘           │  Breed classifier    │         │  Breed classifier    │
  LR = 0.001                        └─────────────────────┘         └─────────────────────┘
                                    LR = 0.0001                      LR = 0.00001
```

### Transfer Learning Results

| Model | Stage 1 (Frozen) | Stage 2 (Partial) | Stage 3 (Full) | Best Stage |
|-------|:-------:|:-------:|:-------:|:----------:|
| AlexNet | 17.5% | 25.2% | **26.6%** | Stage 3 |
| MobileNet | 24.5% | **35.7%** | 35.7% | Stage 2 |
| ResNet-50 | 49.0% | 65.7% | **67.1%** | Stage 3 |

![Transfer Learning Stages](results/graphs/transfer_learning_stages.png)

**Key insight:** Stage 2 and 3 consistently improve over Stage 1. The model needs to adapt its feature detectors to dog-specific patterns, not just rely on general ImageNet features.

---

## 📈 Results & Analysis

### What Do These Numbers Mean?

- **Top-1 Accuracy:** The model's #1 guess is correct
- **Top-5 Accuracy:** The correct answer is somewhere in the model's top 5 guesses
- ResNet-50's **95.8% Top-5** means it almost always has the right breed in its top 5!

### Why ResNet-50 Wins

1. **Skip connections** let gradients flow freely → trains better
2. **50 layers** can learn very fine-grained features (fur texture differences)
3. **25M parameters** — sweet spot between too few (can't learn) and too many (overfits)

### Why MobileNet Beats AlexNet Despite Being 24× Smaller

| | AlexNet | MobileNet |
|--|---------|-----------|
| Params | 57M | 2.4M |
| Accuracy | 26.6% | 35.7% |
| Design year | 2012 | 2018 |

MobileNet uses **depthwise separable convolutions** — a more efficient way to process images. Six years of research produced architectures that are both smaller AND more accurate.

---

## 🦁 Experiment: What Dog Breed Is This Animal?

We fed images of 10 non-dog animals through our trained ResNet-50 and asked: *"What dog breed is this?"*

The model was trained ONLY on dog breeds, so it has to pick the closest match from 120 dog breeds. The results reveal what visual features the CNN actually learned!

| Animal | Predicted Dog Breed | Confidence | Why It Makes Sense |
|:------:|:------------------:|:----------:|:-------------------|
| 🐻 Bear | **Collie** | 10.4% | Fluffy fur, similar face shape |
| 🐱 Cat | **Pomeranian** | 8.9% | Small face, fluffy fur, pointed ears |
| 🐄 Cow | **English Foxhound** | 8.1% | Spotted pattern, similar body build |
| 🫏 Donkey | **English Foxhound** | 2.3% | Long face, similar proportions |
| 🦊 Fox | **Dingo** | 7.3% | Wild canine! Very similar features |
| 🐴 Horse | **Saluki** | 39.2% | Long legs, slender build, elegant posture |
| 🦁 Lion | **Dhole** | 4.9% | Wild canine face shape, tawny color |
| 🐰 Rabbit | **Pomeranian** | 3.2% | Small, fluffy, round face |
| 🐺 Wolf | **African Hunting Dog** | 3.7% | Wild canine — closest real match! |
| 🦓 Zebra | **German Short-haired Pointer** | 3.6% | Pattern recognition, spotted coat |

### Key Observations

- **Fox → Dingo:** The CNN correctly identified the fox as closest to a wild canine. Impressive!
- **Horse → Saluki (39.2%!):** The highest confidence non-dog prediction. Salukis are tall, slender, elegant — just like horses. The CNN learned body shape, not just fur.
- **Wolf → African Hunting Dog:** Another wild canine match — the CNN sees the family resemblance!
- **Confidence is lower** for non-dogs (2-39%) vs real dogs (typically 50-90%), showing the model "knows" these aren't really dogs.

---

## 🧑‍🤝‍🧑 Experiment: Which Dog Breed Are You?

We fed 20 human face photos through the dog breed model. Remember: the model has NEVER seen a human face — it can only pick from 120 dog breeds!

| Person | Predicted Breed | Confidence | 2nd Choice | 3rd Choice |
|:------:|:--------------:|:----------:|:----------:|:----------:|
| Person 01 | **Toy Poodle** | 5.8% | Italian Greyhound | Miniature Poodle |
| Person 02 | **Toy Poodle** | 4.5% | Staffordshire Bull Terrier | Italian Greyhound |
| Person 03 | **Toy Poodle** | 3.7% | Italian Greyhound | Staffordshire Bull Terrier |
| Person 04 | **Toy Poodle** | 6.6% | Brittany Spaniel | Weimaraner |
| Person 05 | **Toy Poodle** | 4.2% | Italian Greyhound | Gordon Setter |
| Person 06 | **Toy Poodle** | 5.5% | Brittany Spaniel | Weimaraner |
| Person 07 | **Toy Poodle** | 5.9% | Sussex Spaniel | Miniature Poodle |
| Person 08 | **Toy Poodle** | 6.1% | Italian Greyhound | Pug |
| **Person 09** | **Affenpinscher** | **17.5%** | Lhasa Apso | Miniature Poodle |
| Person 10 | **Toy Poodle** | 4.2% | Weimaraner | Brittany Spaniel |
| **Person 11** | **Italian Greyhound** | **20.2%** | Komondor | Gordon Setter |
| **Person 12** | **Komondor** | **20.3%** | Staffordshire Bull Terrier | Gordon Setter |
| Person 13 | **Italian Greyhound** | 6.0% | Gordon Setter | Toy Poodle |
| Person 14 | **Staffordshire Bull Terrier** | 4.7% | Toy Poodle | Chihuahua |
| Person 15 | **Chihuahua** | 7.3% | Staffordshire Bull Terrier | Pug |
| Person 16 | **Italian Greyhound** | 5.3% | Toy Poodle | Lhasa Apso |
| Person 17 | **Toy Poodle** | 10.7% | Miniature Poodle | Italian Greyhound |
| Person 18 | **Toy Poodle** | 5.4% | Italian Greyhound | Staffordshire Bull Terrier |
| Person 19 | **Toy Poodle** | 6.9% | Sussex Spaniel | Komondor |
| Person 20 | **Italian Greyhound** | 8.0% | Miniature Poodle | Toy Poodle |

### Most Interesting Matches

**Person 09 → Affenpinscher (17.5%)** — The Affenpinscher is literally called the "Monkey Dog" because of its human-like face! The CNN detected the similarity.

**Person 12 → Komondor (20.3%)** — The Komondor has long, corded white hair resembling dreadlocks. This person likely has long or curly hair.

**Person 11 → Italian Greyhound (20.2%)** — Slim facial features, elegant proportions — the CNN maps these to the slender Italian Greyhound.

### What the CNN "Sees" in Human Faces

- **Most faces → Toy Poodle:** Curly/wavy hair texture maps to poodle-like fur
- **Slim faces → Italian Greyhound:** The CNN associates slender features with this elegant breed
- **Round faces → Pug/Chihuahua:** Compact facial features trigger these small-breed detectors
- **Confidence is very low (3-20%):** The model correctly senses these are NOT dogs — it's unsure

This reveals that CNNs learn **texture and shape features**, not semantic concepts like "this is a dog" or "this is a human."

---

## 💡 Insights & Conclusions

### What We Learned

1. **Transfer learning is essential** for small datasets. Without it, accuracy is near random (1.4% vs 67.1%).

2. **Architecture innovation matters more than brute force.** MobileNet (2.4M params, 35.7%) beats AlexNet (57M params, 26.6%) — proving that clever design outperforms raw size.

3. **ResNet's skip connections are transformative.** They enabled training 50-layer networks where previous architectures would fail from vanishing gradients.

4. **Top-5 accuracy is the real story.** ResNet-50's 95.8% means the correct breed is almost always in the model's top 5 guesses. Many "mistakes" are between visually similar breeds.

5. **CNNs learn features, not concepts.** The animal and celebrity experiments show the model matches based on visual texture and shape, not understanding what a "dog" is.

### What Surprised Us

- A horse is a Saluki (39.2% confidence!) — the slender, long-legged body shape
- A fox is a Dingo (7.3%) — the CNN found the closest wild canine relative
- Most human faces are "Toy Poodles" — curly hair texture dominates the prediction
- MobileNet, designed for phones, beats AlexNet, designed for GPUs

### Limitations

- 10% data subset gives preliminary results — full Colab training will be more accurate
- 120 classes with ~80 images each is challenging — some breeds look nearly identical
- Celebrity experiment uses stock photos, not actual celebrities (for copyright reasons)
- The model can only choose from 120 breeds — it can't say "this is not a dog"

### What Architecture Would We Recommend?

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| **Best accuracy** | ResNet-50 | Skip connections enable deep learning |
| **Mobile app** | MobileNet | 10× fewer params, fast inference |
| **Learning CNNs** | Simple CNN | Build from scratch, understand every layer |
| **Research baseline** | VGG-16 | Simple, well-understood, widely cited |

---

## 🚀 How to Run This Project

### Option A: Google Colab (Recommended — GPU)

1. Open the [Colab Notebook](https://colab.research.google.com)
2. Upload `notebooks/dog_breed_classifier.ipynb`
3. Runtime → Change runtime type → **A100 GPU**
4. Run all cells (~1-2 hours)

### Option B: WSL / Local (CPU, 10% data)

```bash
cd /mnt/c/2025AIDEV/L41
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m src.data.download
python -m src.data.organize
python run_training.py
```

### Option C: Windows PowerShell

```powershell
cd C:\2025AIDEV\L41
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
python run_training.py
```

---

## 📁 Project Structure

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
│   │   ├── simple_cnn.py       # Custom baseline (221K params)
│   │   ├── alexnet.py          # AlexNet 2012 (57M params)
│   │   ├── vgg.py              # VGG-16 2014 (138M params)
│   │   ├── inception.py        # GoogLeNet 2014 (25M params)
│   │   ├── resnet.py           # ResNet-50 2015 (25M params)
│   │   └── mobilenet.py        # MobileNet 2018 (2.4M params)
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py          # Training loop + early stopping
│   │   ├── transfer_learning.py # 3-stage TL pipeline
│   │   └── evaluate.py         # Metrics + confusion matrix
│   ├── inference/
│   │   └── predict.py          # predict_dog_breed() function
│   ├── experiments/            # Animal & celebrity experiments
│   └── visualization/          # Plots, galleries
├── notebooks/
│   └── dog_breed_classifier.ipynb  # 15-cell Colab notebook
├── data/                       # Dataset (gitignored)
├── results/
│   ├── graphs/                 # All plots (committed)
│   ├── tables/                 # CSV results
│   └── models/                 # Saved weights (gitignored)
├── docs/
│   ├── PRD.md                  # Product Requirements
│   └── tasks.json              # 68 tasks, 12 phases
├── run_training.py             # Main training script
├── requirements.txt
└── README.md                   # This file
```

---

## 📚 References

- **AlexNet:** Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural Networks" (2012)
- **VGG:** Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition" (2014)
- **GoogLeNet:** Szegedy et al., "Going Deeper with Convolutions" (2014)
- **ResNet:** He et al., "Deep Residual Learning for Image Recognition" (2015)
- **MobileNet:** Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (2018)
- **Dataset:** [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)

---

*This README will be updated with final Colab results when full GPU training completes.*

*Built with PyTorch, trained on Kaggle Dog Breed Identification dataset (120 breeds, ~10K images).*
