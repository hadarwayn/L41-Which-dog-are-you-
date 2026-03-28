# PRD - Dog Breed Classification & "Human-to-Dog Similarity" Experiment

**Project Name:** Dog Breed Classification & Human-to-Dog Similarity
**Version:** 1.2
**Date:** 2026-03-28 (Updated: 2026-03-28)
**Author:** Hadar Wayne
**Course:** Deep Learning & AI (Dr. Yoram Segal)
**Status:** Draft - Pending Approval

---

## 1. Executive Summary

Build a deep learning image classification system that:
1. Classifies ~120 dog breeds using multiple CNN architectures
2. Compares performance across architectures (accuracy, speed, complexity)
3. Tests trained models on non-dog images (humans, other animals) as a fun experiment
4. Matches famous people to dog breeds based on model predictions

This is an **educational project first and foremost**. The README must serve as a **complete, self-contained learning document** that:
- A **12-year-old** can follow and understand (simple language, real-world analogies, visual diagrams)
- A **neural network expert** finds valuable (clear comparisons, precise metrics, architectural insights, well-organized results)

The key is making the complex simple WITHOUT losing depth — diagrams, examples, comparisons, and results tell the story.

---

## 2. Learning Objectives

Based on course lectures L30-L40 (Dr. Yoram Segal), this project must demonstrate mastery of:

### 2.1 Core CNN Concepts
- **Convolution** — kernel/filter sliding over images, dot product, feature maps
- **Padding** — preserving spatial dimensions (same vs valid)
- **Stride** — step size of the kernel
- **Pooling** — Max Pooling (retains strongest activations), Average Pooling, Global Average Pooling
- **Activation Functions** — ReLU (primary), Leaky ReLU, Sigmoid, Softmax (output layer)
- **Flatten** — converting 2D feature maps to 1D vector for classification
- **Fully Connected (Dense) layers** — final classification layers
- **Softmax + Cross-Entropy Loss** — output probability distribution + loss function for multi-class
- **Backpropagation & Gradient Descent** — weight updates, Adam optimizer
- **Batch Normalization** — internal normalization to stabilize training
- **Dropout** — regularization by randomly deactivating neurons
- **Data Augmentation** — rotation, flip, zoom, shift to expand dataset
- **Weight Sharing** — CNN kernels share weights across the image (fewer params than FC)

### 2.2 Transfer Learning (from lecture L37/L40)
- Using pre-trained ImageNet weights as starting point
- **Feature Extraction** — freeze backbone, train only classifier head
- **Fine-Tuning** — unfreeze top layers for domain adaptation
- Why it works: early layers learn universal features (edges, textures), deeper layers learn task-specific features

### 2.3 Architecture Evolution (from lecture L40 - VGG/ResNet comparison)
The project must implement and explain the historical progression:

| Year | Architecture | Key Innovation | Layers | Params |
|------|-------------|----------------|--------|--------|
| 1998 | **LeNet-5** | First practical CNN | 5 | ~60K |
| 2012 | **AlexNet** | GPU training, ReLU, Dropout | 8 | ~60M |
| 2014 | **VGG-16/19** | Only 3x3 filters, deep & uniform | 16/19 | ~138M |
| 2014 | **GoogLeNet/Inception** | Parallel branches (Inception module), 1x1 conv | 22 | ~5M |
| 2015 | **ResNet-50** | Skip connections (residual learning) | 50 | ~25M |
| 2017+ | **MobileNet** | Depthwise separable convolution, mobile-friendly | — | ~3.4M |

---

## 3. Dataset

### 3.1 Primary Dataset
**Source:** Kaggle - Dog Breed Identification
**URL:** https://www.kaggle.com/c/dog-breed-identification

**Contents:**
- ~10,222 labeled training images
- ~10,357 unlabeled test images
- `labels.csv` — mapping of image ID to breed name
- ~120 unique dog breeds

### 3.2 Additional Test Data (Post-Training — Inference Only)

After all models are trained, collect and run these through the trained NN:

**A) 100 Famous People Images:**
- Diverse set: actors, musicians, athletes, politicians, scientists
- Face-focused images (clear, front-facing preferred)
- Collected from public/fair-use sources
- Purpose: "Which dog breed does this person most resemble?"

**B) 10 Non-Dog Animal Types (2-3 images each, ~25 total):**
- Horse, Zebra, Cat, Donkey, Rabbit, Fox, Wolf, Bear, Lion, Cow
- Clear, single-animal images
- Purpose: "What dog breed does the NN think this animal is?"

**C) Inference Process (same for all non-dog images):**
1. Preprocess image (resize to 224x224, normalize with ImageNet stats)
2. Feed through the **best trained model** (from Experiment 1)
3. Get softmax output → probability distribution over 120 breeds
4. Extract top-3 predicted breeds with confidence percentages
5. Display: input image ↔ predicted breed name + sample breed image + confidence
6. Analyze: confidence levels, patterns, funny/surprising results

---

## 4. Data Pipeline

### 4.1 Download & Extract
1. Download from Kaggle (API or manual)
2. Extract `train.zip` and `test.zip`
3. Parse `labels.csv`

### 4.2 Organize into Folder Structure
```
data/
├── raw/                    # Original downloaded files
│   ├── train/              # Raw training images
│   ├── test/               # Raw test images (unlabeled)
│   └── labels.csv
├── processed/
│   ├── train/              # 80% of labeled data
│   │   ├── affenpinscher/
│   │   │   ├── img001.jpg
│   │   │   └── ...
│   │   ├── afghan_hound/
│   │   └── ... (120 breed folders)
│   ├── val/                # 20% of labeled data
│   │   ├── affenpinscher/
│   │   └── ...
│   └── test/               # Non-dog images for experiments
│       ├── humans/
│       └── animals/
```

### 4.3 Compatibility
Must work with:
- **PyTorch:** `torchvision.datasets.ImageFolder` + `DataLoader`
- **OR TensorFlow:** `tf.keras.utils.image_dataset_from_directory`

Framework choice: **PyTorch** (primary) or **TensorFlow/Keras** (alternative)

---

## 5. Data Quality & Preprocessing (MANDATORY — Run Before Any Training)

**CRITICAL:** All data preparation steps MUST complete before any model training begins. The dataset must be balanced and standardized first.

### 5.1 Dataset Analysis (mandatory — first step)
- Class distribution histogram (samples per breed)
- Identify min/max/mean/median samples per class
- Flag breeds with < 50 samples
- Report total usable images
- Identify and remove corrupted/unreadable images

### 5.2 Resolution Standardization (mandatory)
All images MUST be resized to a **uniform, reasonable resolution** before training:
- **Standard size:** 224x224 pixels (for VGG, ResNet, AlexNet, MobileNet, Simple CNN)
- **Inception size:** 299x299 pixels (for GoogLeNet/Inception only)
- Original Kaggle images vary in size — standardize them all during preprocessing
- Save resized images to `data/processed/` to avoid re-processing each run
- Use high-quality resampling (Lanczos/bilinear) to preserve detail

### 5.3 Class Balancing (mandatory)
The dataset groups MUST be balanced before training. Implement **both**:

**A) Class Weights in Loss Function:**
- Compute weight per breed: `weight[i] = total_samples / (num_classes * count[i])`
- Pass weights to `CrossEntropyLoss(weight=class_weights)`
- Ensures rare breeds have proportionally higher impact on loss

**B) Data Augmentation for Underrepresented Classes:**
- Breeds with fewer samples get **more aggressive augmentation**
- Target: each breed has roughly equal effective training representation
- Augmentation techniques:
  - Random horizontal flip
  - Random rotation (±15°)
  - Random affine transformations
  - Color jitter (brightness, contrast, saturation)
  - Random erasing (cutout)

**C) Optional — Downsampling/Upsampling:**
- Cap overrepresented breeds at max N samples
- Duplicate underrepresented breeds to min N samples
- Use `WeightedRandomSampler` for balanced batch sampling

### 5.4 Image Preprocessing Pipeline
Applied to ALL images uniformly:
1. **Resize** to 224x224 (or 299x299 for Inception)
2. **Normalize** using ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
3. **Data Augmentation** (training only — see 5.3B)
4. **Minimum dataset size:** at least 3,000 images used for training

### 5.5 Train/Validation Split
- 80% training / 20% validation (**stratified by breed** — same proportion per class)
- No data leakage — split before any preprocessing/augmentation
- Validation set uses only resize + normalize (no augmentation)

### 5.6 10% Subset for Local Testing (WSL)
- Create a stratified 10% sample of the full training set
- Same breed distribution as full set (just fewer images per breed)
- Used by Claude Code for autonomous testing in WSL terminal
- Stored in `data/processed/train_subset_10pct/`

---

## 6. Model Architectures

### 6.1 Models to Implement

#### Model 1: Simple CNN (Baseline)
- Custom-built from scratch
- Architecture: Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Flatten → Dense → Dropout → Dense(120, softmax)
- Purpose: establish baseline, demonstrate CNN fundamentals
- Demonstrates: convolution, pooling, activation, fully connected layers

#### Model 2: AlexNet (Historical)
- 8 layers (5 conv + 3 FC)
- Key innovations: ReLU activation, Dropout, GPU training
- Large filters (11x11, 5x5) in early layers
- Transfer learning from ImageNet weights (if available) or custom implementation

#### Model 3: VGG-16
- 16 weight layers, only 3x3 convolutions
- Very uniform architecture: [Conv3x3, Conv3x3, MaxPool] repeated
- ~138M parameters — demonstrates "depth with simplicity"
- Pre-trained on ImageNet (torchvision.models.vgg16)

#### Model 4: GoogLeNet / Inception v3
- Inception modules: parallel 1x1, 3x3, 5x5 convolutions + pooling
- 1x1 convolutions for dimensionality reduction
- ~5M parameters (very efficient vs VGG)
- Input size: 299x299
- Pre-trained on ImageNet

#### Model 5: ResNet-50
- 50 layers with skip connections (residual blocks)
- Bottleneck blocks: 1x1 → 3x3 → 1x1 convolutions
- Solves vanishing gradient problem
- ~25M parameters
- Pre-trained on ImageNet

#### Model 6: MobileNet v2
- Depthwise separable convolutions
- Inverted residuals with linear bottlenecks
- ~3.4M parameters — designed for mobile/edge devices
- Pre-trained on ImageNet

### 6.2 Architecture Components Used (all must appear in at least one model)
- [x] Convolution (Conv2D)
- [x] Max Pooling
- [x] Average Pooling / Global Average Pooling
- [x] ReLU activation
- [x] Softmax output
- [x] Dropout
- [x] Batch Normalization
- [x] Skip / Residual connections (ResNet)
- [x] 1x1 convolutions (Inception, ResNet)
- [x] Depthwise separable convolution (MobileNet)
- [x] Flatten + Fully Connected (Dense)

---

## 7. Transfer Learning — 3 Stages (Mandatory)

For each pre-trained model (VGG-16, ResNet-50, Inception, MobileNet):

### Stage 1: Feature Extraction
- Load pre-trained ImageNet weights
- **Freeze ALL backbone layers** (set `requires_grad = False`)
- Replace final classifier with: `Dense(256, ReLU) → Dropout(0.5) → Dense(120, Softmax)`
- Train only the new classifier head
- Epochs: 10-15
- Learning rate: 1e-3

### Stage 2: Partial Fine-Tuning
- **Unfreeze top 20-30% of backbone layers**
- Keep early layers frozen (they learn universal features)
- Lower learning rate: 1e-4 to 1e-5
- Epochs: 10-15

### Stage 3: Full Fine-Tuning
- **Unfreeze entire network**
- Very low learning rate: 1e-5 to 1e-6
- Use learning rate scheduling (ReduceLROnPlateau)
- Epochs: 10-20
- Risk of overfitting — monitor validation loss carefully

### Transfer Learning Comparison Table (to generate)
| Model | Stage 1 Acc | Stage 2 Acc | Stage 3 Acc | Best Stage |
|-------|-----------|-----------|-----------|------------|
| VGG-16 | | | | |
| ResNet-50 | | | | |
| Inception | | | | |
| MobileNet | | | | |

---

## 8. Training Configuration

### 8.1 Hyperparameters
- **Optimizer:** Adam (with adaptive learning rate)
- **Loss Function:** CrossEntropyLoss (multi-class)
- **Batch Size:** 32 (adjustable based on GPU memory)
- **Epochs:** 10-20 per stage
- **Learning Rate:** varies by stage (see Section 7)
- **LR Scheduler:** ReduceLROnPlateau (patience=3, factor=0.1)
- **Early Stopping:** patience=5 on validation loss

### 8.2 Dual Execution Environments

This project runs in **two environments** with different purposes:

#### Environment A: WSL Terminal (Local — Claude Code runs autonomously)
- **Purpose:** Code validation, quick testing, initial results for README
- **Data:** 10% subset of the full dataset (stratified sample — same breed distribution)
- **Who runs it:** Claude Code (the AI assistant) runs this autonomously
- **Hardware:** CPU or local GPU (if available)
- **Output:** Preliminary results, code verification, initial README content
- **Workflow:**
  1. Claude Code writes all src/ code
  2. Claude Code downloads/prepares 10% data subset
  3. Claude Code runs training on all 6 architectures (fewer epochs, smaller data)
  4. Claude Code generates preliminary graphs and tables
  5. Claude Code writes README with these initial results
  6. Results labeled as "Preliminary (10% data)" in README

#### Environment B: Google Colab (Full Training — User runs manually)
- **Purpose:** Full dataset training, final results, celebrity/animal experiments
- **Colab URL:** https://colab.research.google.com/drive/1Rvgw3hEfrl53GqFkeXpQUZZYdSajRmpJ
- **Who runs it:** The user (Hadar) runs this manually
- **Hardware:** Colab GPU (T4/A100) — user is willing to pay for Colab Pro for better results
- **Storage:** Google Drive for data, checkpoints, and results
- **Output:** Final production results, full experiments
- **Workflow:**
  1. User opens the Colab notebook
  2. Notebook mounts Google Drive, downloads full dataset
  3. Full training on all architectures (all 3 transfer learning stages)
  4. Runs Experiments 2 & 3 (non-dog images, celebrity matching)
  5. Generates final graphs, tables, and gallery
  6. **Results Export Section:** A dedicated cell at the end produces a JSON/text summary that the user copies back to Claude Code
  7. Claude Code updates README with final (full) results

#### Results Flow
```
┌──────────────────────┐     ┌──────────────────────┐
│  WSL Terminal        │     │  Google Colab         │
│  (Claude runs)       │     │  (User runs)          │
│                      │     │                       │
│  10% data subset     │     │  100% full dataset    │
│  Quick validation    │     │  Full training (GPU)  │
│  Preliminary results │     │  Celebrity experiment │
│  Initial README      │     │  Final results        │
│         │            │     │         │             │
│         ▼            │     │         ▼             │
│  README v1           │     │  Export results cell   │
│  (preliminary)       │     │  (copy-paste to Claude)│
└──────────────────────┘     └──────────────────────┘
                                       │
                                       ▼
                              README v2 (final)
```

### 8.3 Colab Notebook Structure
The notebook must include a **"Results Export" section** at the end with:
```python
# === RESULTS EXPORT SECTION ===
# Run this cell after all training is complete.
# Copy the output and paste it to Claude Code to update README.md

results_export = {
    "model_comparison": { ... },      # Final accuracy/loss per model
    "transfer_learning": { ... },     # Stage 1/2/3 results per model
    "best_model": "...",              # Name of best performing model
    "training_times": { ... },        # Time per model
    "non_dog_experiment": { ... },    # Top predictions for animals
    "celebrity_experiment": { ... },  # Top matches for celebrities
}
print(json.dumps(results_export, indent=2))
```

### 8.4 Colab Pro Recommendation
The user is willing to pay for Colab Pro. Recommended setup:
- **Colab Pro** ($9.99/month) — T4 GPU, longer runtimes
- **Colab Pro+** ($49.99/month) — A100 GPU, background execution (recommended if training all 6 models)
- Use **high-RAM runtime** for VGG-16 (138M params)
- Enable **GPU acceleration** in Runtime → Change runtime type

---

## 9. Experiments

### 9.1 Experiment 1: Architecture Comparison
**Goal:** Compare all 6 architectures on dog breed classification

**Metrics to record per model:**
- Training accuracy (per epoch)
- Validation accuracy (per epoch)
- Training loss (per epoch)
- Validation loss (per epoch)
- Total training time
- Number of parameters
- Inference time per image
- Top-1 and Top-5 accuracy

**Output:**
- Accuracy vs Epochs graph (all models overlaid)
- Loss vs Epochs graph (all models overlaid)
- Comparison table (Model | Accuracy | Loss | Time | Params | Top-5)
- Confusion matrix for best model
- **Example predictions grid** showing 3 categories:
  1. **Correct classifications** (model got it right, green border)
  2. **Wrong classifications** (model got it wrong, red border — show predicted vs actual)
  3. **Funny/interesting results** (humans/animals as dogs, yellow border)

### 9.2 Experiment 2: Non-Dog Animal Predictions
**Goal:** Feed images of 10 non-dog animal types through the trained NN to see what dog breed it predicts

**Input: 10 Animal Types (2-3 images each, ~25 images total):**
1. Horse
2. Zebra
3. Cat
4. Donkey
5. Rabbit
6. Fox
7. Wolf
8. Bear
9. Lion
10. Cow

**Step-by-Step Inference Pipeline:**
```
For each animal image:
  1. Load image
  2. Resize to 224x224
  3. Normalize (ImageNet mean/std)
  4. Feed through best trained model
  5. Apply softmax → get probability distribution over 120 breeds
  6. Extract top-3 predicted breeds + confidence percentages
  7. Save: {animal_type, image_name, breed_1, conf_1, breed_2, conf_2, breed_3, conf_3}
```

**Output:**
- Table: Animal | Image | Top-1 Breed | Confidence | Top-2 Breed | Top-3 Breed
- Side-by-side image: input animal ↔ predicted dog breed sample photo
- Confidence distribution chart (how "sure" is the model about non-dog inputs?)

**Analysis Questions:**
- Are confidence scores lower for non-dogs vs actual dogs? (expected: yes)
- Does a wolf get predicted as a Husky or German Shepherd? (expected: likely)
- Does a zebra get predicted as a Dalmatian? (pattern similarity)
- Do similar animals (fox/wolf) get similar breed predictions?
- What does this tell us about what features the CNN learned?

### 9.3 Experiment 3: Famous People → Dog Breed Matching ("Funny AI")
**Goal:** The fun experiment — run 100 famous people through the dog breed NN and see which breed each person "is"

**Input: ~100 Famous People (diverse set):**
- ~30 actors/actresses (e.g., Brad Pitt, Meryl Streep, Keanu Reeves...)
- ~20 musicians (e.g., Beyonce, Ed Sheeran, Billie Eilish...)
- ~15 athletes (e.g., Messi, Serena Williams, LeBron James...)
- ~15 politicians/leaders (e.g., Obama, Merkel, historical figures...)
- ~10 scientists/tech (e.g., Einstein, Elon Musk, Ada Lovelace...)
- ~10 other (comedians, YouTubers, fictional characters for fun)

**Step-by-Step Inference Pipeline (same as Experiment 2):**
```
For each celebrity image:
  1. Load face image
  2. Resize to 224x224
  3. Normalize (ImageNet mean/std)
  4. Feed through best trained model (e.g., ResNet-50)
  5. Apply softmax → probability distribution over 120 dog breeds
  6. Extract top-3 predicted breeds + confidence percentages
  7. Find a sample image of the predicted breed from the training set
  8. Save: {person_name, breed_1, conf_1, breed_2, conf_2, breed_3, conf_3}
```

**Output:**
- **Celebrity-Dog Gallery:** Side-by-side grid showing:
  - Celebrity photo → Top predicted dog breed photo → Breed name → Confidence %
- **Full results table:** Person | Breed 1 (conf%) | Breed 2 (conf%) | Breed 3 (conf%)
- **"Top 10 Funniest Matches"** — hand-picked most amusing results
- **Confidence analysis:** histogram of confidence scores across all celebrities

**Analysis Questions:**
- Are confidence scores generally low? (expected: yes — these aren't dogs)
- Do people with similar features (hair color, face shape) get similar breeds?
- Do bald people get matched to hairless breeds?
- What does this reveal about WHAT FEATURES the CNN actually learned?
- Discussion: the model was trained to find "dog-ness" — it maps human features to the closest dog feature space

### 9.4 Post-Experiment: Inference Module
**Goal:** Create a reusable inference function that takes ANY image and returns the top-3 predicted dog breeds

```python
def predict_dog_breed(image_path: str, model: nn.Module) -> list[dict]:
    """
    Feed any image through the trained model.

    Returns: [
        {"breed": "golden_retriever", "confidence": 0.73},
        {"breed": "labrador_retriever", "confidence": 0.12},
        {"breed": "cocker_spaniel", "confidence": 0.05}
    ]
    """
```

This function is used by both Experiment 2 and Experiment 3, and can be used on any new image.

---

## 10. Output Requirements

### 10.1 Visualizations (mandatory)
1. **Accuracy vs Epochs** — line plot, one line per model (all 6 overlaid)
2. **Loss vs Epochs** — line plot, one line per model (all 6 overlaid)
3. **Confusion Matrix** — heatmap for best model (top-20 most confused breeds)
4. **Class Distribution** — bar chart of samples per breed (before and after balancing)
5. **Example Predictions Grid** — 3 categories:
   - Correct predictions (green border) — at least 6 examples
   - Wrong predictions (red border) — at least 6 examples
   - Funny results (yellow border) — humans/animals as dogs, at least 6 examples
6. **Transfer Learning Stages** — bar chart comparing Stage 1/2/3 accuracy per model
7. **Architecture Comparison Bar Chart** — final accuracy side-by-side for all 6 models
8. **Animal Experiment Gallery** — 10 animals → predicted breeds, side-by-side
9. **Celebrity-Dog Gallery** — top-20 funniest celebrity → breed matches, side-by-side
10. **Confidence Distribution** — histogram of confidence scores (dogs vs non-dogs vs humans)

### 10.2 Comparison Tables
| Model | Top-1 Acc | Top-5 Acc | Params | Train Time | Inference (ms) |
|-------|----------|----------|--------|-----------|---------------|
| Simple CNN | | | | | |
| AlexNet | | | | | |
| VGG-16 | | | | | |
| Inception | | | | | |
| ResNet-50 | | | | | |
| MobileNet | | | | | |

### 10.3 Results Directory
```
results/
├── graphs/
│   ├── accuracy_comparison.png
│   ├── loss_comparison.png
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   ├── transfer_learning_stages.png
│   └── sample_predictions.png
├── tables/
│   ├── model_comparison.csv
│   └── transfer_learning_comparison.csv
├── experiments/
│   ├── non_dog_predictions.csv
│   ├── celebrity_dog_matches.csv
│   └── celebrity_gallery/
│       ├── match_001.png
│       └── ...
└── models/
    ├── best_simple_cnn.pth
    ├── best_alexnet.pth
    ├── best_vgg16.pth
    ├── best_inception.pth
    ├── best_resnet50.pth
    └── best_mobilenet.pth
```

---

## 11. README Requirements

The README.md must serve as a **complete, self-contained learning document** with TWO audiences:

**Audience 1 — A 12-year-old student:** Can read it and understand what CNNs are, how they work, and what the project does. Uses simple language, everyday analogies, and visual diagrams.

**Audience 2 — A neural network expert:** Finds the README valuable because of clear architecture comparisons, precise metrics, well-organized results, and insightful analysis. No hand-waving — real data, real numbers.

**The magic formula:** Explain concepts simply with analogies → THEN show the precise technical details → THEN show results and data → THEN provide insight.

### 11.1 CNN Concepts Section (Educational Foundation)
Each concept explained with:
- **Simple analogy** a 12-year-old would understand
- **ASCII/text diagram** showing how it works visually
- **Technical detail** (formula or precise description)
- **Example** from this project

Topics to cover:
- What is a neural network? (brain analogy)
- What is a CNN? (why it's special for images)
- What is convolution? (sliding magnifying glass analogy)
- What is a kernel/filter? (pattern detector)
- What does pooling do? (shrinking while keeping important info)
- What is ReLU? (keeping only positive signals)
- What is softmax? (turning numbers into percentages that add to 100%)
- What is a loss function? (how wrong is the model?)
- What is backpropagation? (learning from mistakes)
- What is an epoch? (one full study session)
- What is overfitting? (memorizing answers vs understanding)
- What is data augmentation? (creating variety from existing data)

### 11.2 Architecture Explanations
For each of the 6 architectures, include:
- **What is it?** — one-paragraph summary
- **When was it created and why?** — historical context
- **How does it work?** — key mechanism with ASCII/visual diagram
- **What makes it special?** — the key innovation
- **When to use it?** — practical real-world use cases
- **Analogy** — comparison a 12-year-old would understand
- **Diagram** — architecture visualization (ASCII blocks or described visual)

### 11.3 Architecture Comparison Table (Expert-Level)
Detailed comparison with real data from this project:

| Architecture | Year | Layers | Params | Key Innovation | Our Accuracy | Training Time | Best For |
|-------------|------|--------|--------|----------------|-------------|--------------|---------|
| Simple CNN | — | 9 | ~1M | Baseline | X% | Xm | Learning |
| AlexNet | 2012 | 8 | 60M | ReLU+Dropout | X% | Xm | Historical |
| VGG-16 | 2014 | 16 | 138M | 3x3 only | X% | Xm | Depth |
| Inception | 2014 | 22 | 5M | Parallel paths | X% | Xm | Efficiency |
| ResNet-50 | 2015 | 50 | 25M | Skip connections | X% | Xm | Best overall |
| MobileNet | 2017 | — | 3.4M | Depthwise sep. | X% | Xm | Mobile |

### 11.4 Transfer Learning Section
- **Analogy:** "You already know how to ride a bicycle → learning to ride a motorcycle is faster than learning from scratch"
- Why transfer learning works (with diagram)
- The 3 stages explained simply:
  - Stage 1: "Use a pre-trained brain, just change the decision-making part"
  - Stage 2: "Also re-learn some of the advanced pattern recognition"
  - Stage 3: "Re-learn everything, but carefully"
- Results: which stage was best for each model (with bar chart)

### 11.5 Results & Analysis (Two-Pass Approach)
**Pass 1 — Preliminary (Claude Code writes from WSL 10% test):**
- Initial accuracy/loss results labeled as "Preliminary (10% subset)"
- Quick architecture comparison
- Validates that all code works correctly

**Pass 2 — Final (User pastes Colab full-run results):**
- Claude Code updates README with final results from full Colab training
- Which model performed best and **why** (not just numbers — explanation)
- Transfer learning: which stage gave best results and why
- Surprising findings
- Model limitations
- Full celebrity and animal experiment results

### 11.6 Example Predictions Section (3 categories)
Display a grid of example predictions in 3 categories:
1. **Correct predictions** (green) — model got the breed right, show confidence
2. **Wrong predictions** (red) — model got it wrong, show what it predicted vs actual
3. **Funny/interesting predictions** (yellow) — humans or animals predicted as dogs

### 11.7 Animal Experiment Section
- Show each of the 10 animal types and what breed the NN predicted
- Side-by-side: animal photo → predicted dog breed photo
- Analysis: why did the model choose this breed? (feature similarities)

### 11.8 Celebrity Experiment Section
- Methodology (step-by-step how images go through the NN)
- **Gallery of matches** (celebrity ↔ predicted breed, with confidence)
- **Top 10 Funniest Matches** highlight
- Why the model makes these predictions (what features does it "see"?)
- What this teaches us about how CNNs process images

### 11.9 Insights & Conclusions
- What surprised us
- What we learned about CNN architectures
- Which architecture would we recommend for different scenarios
- Limitations of this approach
- Ideas for future improvements

---

## 12. Project Structure

```
L41/
├── docs/
│   ├── PRD.md                          # This document
│   ├── tasks.json                      # Task breakdown
│   └── L30-40_LecturesesSummary/       # Lecture reference PDFs
├── src/
│   ├── __init__.py
│   ├── config.py                       # Hyperparameters, paths, constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py                 # Kaggle API download
│   │   ├── organize.py                 # Reorganize into breed folders
│   │   ├── dataset.py                  # PyTorch Dataset/DataLoader
│   │   ├── augmentation.py             # Data augmentation transforms
│   │   └── analysis.py                 # Class distribution analysis
│   ├── models/
│   │   ├── __init__.py
│   │   ├── simple_cnn.py               # Custom baseline CNN
│   │   ├── alexnet.py                  # AlexNet implementation
│   │   ├── vgg.py                      # VGG-16 with transfer learning
│   │   ├── inception.py                # GoogLeNet/Inception
│   │   ├── resnet.py                   # ResNet-50
│   │   └── mobilenet.py               # MobileNet v2
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Training loop, metrics, logging
│   │   ├── transfer_learning.py        # 3-stage transfer learning logic
│   │   └── evaluate.py                 # Evaluation, confusion matrix
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict.py                  # predict_dog_breed(image) → top-3 breeds
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── compare_architectures.py    # Experiment 1
│   │   ├── non_dog_predictions.py      # Experiment 2 (10 animal types)
│   │   └── celebrity_matching.py       # Experiment 3 (100 famous people)
│   └── visualization/
│       ├── __init__.py
│       ├── plots.py                    # Accuracy/loss graphs
│       ├── confusion.py               # Confusion matrix
│       └── gallery.py                 # Celebrity-dog gallery
├── notebooks/
│   └── dog_breed_classifier.ipynb      # Main Colab notebook
├── data/                               # (gitignored, on Google Drive)
│   ├── raw/
│   └── processed/
├── results/
│   ├── graphs/
│   ├── tables/
│   ├── experiments/
│   └── models/
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## 13. Technical Constraints

### 13.1 Environment A — WSL Terminal (Local)
- **OS:** Windows 11 + WSL (Ubuntu)
- **Purpose:** Code development, 10% data testing, preliminary results
- **Python:** 3.10+
- **Virtual Env:** UV (as per course standard)
- **Hardware:** CPU (or local GPU if available)
- **Data path:** `/mnt/c/2025AIDEV/L41/data/`

### 13.2 Environment B — Google Colab
- **URL:** https://colab.research.google.com/drive/1Rvgw3hEfrl53GqFkeXpQUZZYdSajRmpJ
- **Runtime:** GPU (T4 minimum, A100 preferred with Colab Pro/Pro+)
- **Storage:** Google Drive for dataset, checkpoints, and results
- **Budget:** User willing to pay for Colab Pro for better GPU and longer runtimes
- **Access:** If Claude Code needs editor access to the notebook, user will grant it

### 13.3 Shared
- **Language:** Python 3.10+
- **Framework:** PyTorch 2.x (with torchvision) OR TensorFlow 2.x (with Keras)
- **Key Libraries:** matplotlib, seaborn, pandas, numpy, scikit-learn, Pillow, kaggle
- **Notebook:** Single consolidated Colab notebook with all experiments + results export section
- **Local code:** Modular Python files in `src/` (importable from both WSL and notebook)

---

## 14. Success Criteria

### Must Have (P0)
- [ ] Dataset balanced (class weights + augmentation) and resolution standardized before training
- [ ] At least 4 architectures trained and compared (ResNet-50, VGG-16, Inception, MobileNet)
- [ ] Transfer learning with all 3 stages implemented
- [ ] 10% subset tested locally in WSL with preliminary results
- [ ] Full training in Google Colab with final results
- [ ] Results export section in Colab notebook (user copies back to update README)
- [ ] Accuracy vs Epochs and Loss vs Epochs graphs
- [ ] Confusion matrix for best model
- [ ] Comparison table with metrics
- [ ] Reusable inference function: predict_dog_breed(image) → top-3 breeds
- [ ] Non-dog animal experiment: 10 animal types through NN with results (Colab)
- [ ] Celebrity-dog matching: 100 famous people through NN with gallery (Colab)
- [ ] Example predictions grid: correct, wrong, and funny categories
- [ ] README as complete learning document (12yo understandable, expert-valuable)
- [ ] README includes visual diagrams for all CNN concepts and architectures

### Should Have (P1)
- [ ] All 6 architectures (including Simple CNN and AlexNet)
- [ ] Top-5 accuracy metric
- [ ] Per-breed precision/recall analysis
- [ ] Interactive gallery of celebrity matches
- [ ] Class imbalance handling comparison

### Nice to Have (P2)
- [ ] Grad-CAM visualizations (what the model "sees")
- [ ] Model ensemble combining multiple architectures
- [ ] Web demo (Gradio/Streamlit)
- [ ] t-SNE visualization of learned feature spaces

---

## 15. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Colab GPU timeout | High | Save checkpoints frequently to Drive |
| 120 classes → low per-class accuracy | Medium | Use Top-5 accuracy as secondary metric |
| Dataset imbalance | Medium | Class weights + augmentation |
| Large models don't fit GPU memory | Medium | Reduce batch size, use mixed precision |
| Celebrity images → copyright | Low | Use public domain / fair use images, educational purpose |

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **CNN** | Convolutional Neural Network — neural network with convolution layers for spatial data |
| **Kernel/Filter** | Small matrix that slides over the image to detect features |
| **Feature Map** | Output of a convolution operation — highlights detected patterns |
| **Pooling** | Downsampling operation (max or average) to reduce spatial dimensions |
| **ReLU** | Rectified Linear Unit — activation function: max(0, x) |
| **Softmax** | Converts raw scores to probability distribution (sum = 1) |
| **Cross-Entropy** | Loss function measuring distance between predicted and true distributions |
| **Backpropagation** | Algorithm for computing gradients to update weights |
| **Transfer Learning** | Using a model pre-trained on one task as starting point for another |
| **Fine-Tuning** | Unfreezing and retraining some/all layers of a pre-trained model |
| **Skip Connection** | Shortcut that adds input directly to output, bypassing layers (ResNet) |
| **Depthwise Separable Conv** | Factored convolution: spatial + channel-wise (MobileNet) |
| **ImageNet** | Large-scale image dataset (1.2M images, 1000 classes) used for pre-training |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch Size** | Number of samples processed before updating weights |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Data Augmentation** | Artificially expanding dataset with transformations |

---

**END OF PRD**

*Awaiting approval before proceeding to implementation.*
