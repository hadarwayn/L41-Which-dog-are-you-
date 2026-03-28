# Dog Breed Classification — Deep Learning Project

## Operating Mode

You are a **senior AI engineer and mentor** working on an educational deep learning project.
Operate in **autonomous execution mode** for local development (WSL).
Follow the governance pipeline and never skip steps.

**Rules:** See `.claude/rules/` for mandatory frameworks:
- [autonomous-execution.md](rules/autonomous-execution.md) - Autonomous execution policy
- [implementation-workflow.md](rules/implementation-workflow.md) - Phase-by-phase workflow
- [python-ml.md](rules/python-ml.md) - Python & ML code standards
- [testing.md](rules/testing.md) - Testing requirements
- [git.md](rules/git.md) - Git workflow
- [security.md](rules/security.md) - Security & data safety

## Essential Documentation

| Document | Purpose |
|----------|---------|
| [PRD.md](../docs/PRD.md) | Product requirements, architectures, experiments |
| [tasks.json](../docs/tasks.json) | Task breakdown (58 tasks, 12 phases) |
| [PROJECT_GUIDELINES.md](../docs/PROJECT_GUIDELINES.md) | Course standards (UV, structure, README) |

## Project Overview

**Dog Breed Classification & Human-to-Dog Similarity Experiment**
- Classify ~120 dog breeds using 6 CNN architectures
- Compare: Simple CNN, AlexNet, VGG-16, Inception, ResNet-50, MobileNet
- 3-stage transfer learning (freeze → partial → full fine-tune)
- Fun experiment: which dog breed does each famous person resemble?

## Dual Execution Environments

| | WSL Terminal (Local) | Google Colab (GPU) |
|--|---------------------|-------------------|
| **Who runs** | Claude Code (autonomous) | User (Hadar) manually |
| **Data** | 10% stratified subset | 100% full dataset |
| **Purpose** | Code validation, preliminary results | Full training, final results |
| **Hardware** | CPU | GPU (T4/A100, Colab Pro) |
| **Output** | README v1 (preliminary) | Results Export → README v2 (final) |

**Colab URL:** https://colab.research.google.com/drive/1Rvgw3hEfrl53GqFkeXpQUZZYdSajRmpJ

## Project Structure

```
L41/
├── .claude/              # Claude Code configuration
│   ├── CLAUDE.md         # This file
│   ├── settings.json     # Permissions
│   ├── rules/            # Coding guidelines
│   └── skills/           # Reusable workflows
├── docs/
│   ├── PRD.md            # Requirements
│   ├── tasks.json        # Task tracking
│   └── L30-40_LecturesesSummary/  # Course lecture PDFs
├── src/
│   ├── config.py         # Central configuration
│   ├── data/             # Data pipeline (download, organize, augment)
│   ├── models/           # CNN architectures (6 models)
│   ├── training/         # Training loop, transfer learning, evaluation
│   ├── experiments/      # 3 experiments
│   └── visualization/    # Plots, confusion matrix, gallery
├── notebooks/
│   └── dog_breed_classifier.ipynb  # Main Colab notebook
├── data/                 # Dataset (gitignored)
├── results/              # Outputs (graphs, tables, models)
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Framework** | PyTorch 2.x + torchvision | Model training & inference |
| **Data** | Kaggle Dog Breed dataset | ~10K images, 120 breeds |
| **Visualization** | matplotlib, seaborn | Graphs, confusion matrix |
| **Analysis** | pandas, numpy, scikit-learn | Metrics, data analysis |
| **Environment** | UV (virtual env) | Package management |
| **GPU Runtime** | Google Colab (Pro) | Full training |
| **Local Runtime** | WSL (Ubuntu) | Development & 10% testing |

## Code Standards (Quick Reference)

> Detailed rules in `.claude/rules/python-ml.md`

### File & Function Limits
- **Code files**: 200 lines maximum
- **Single function**: 50 lines maximum
- Split large files by functionality

### Python Standards
- Type hints on all functions
- Docstrings on public functions
- snake_case for files, functions, variables
- PascalCase for classes
- Use `config.py` for all paths and hyperparameters — no hardcoded values

### ML-Specific Standards
- Always set random seeds for reproducibility
- Log all hyperparameters before training
- Save model checkpoints after each epoch/stage
- Validate data pipeline before training (sample visualization)
- Use `torch.no_grad()` during evaluation

## Data Preparation (MANDATORY Before Training)

1. **Download** dataset from Kaggle
2. **Organize** into breed folders (train/val split, stratified)
3. **Standardize resolution** — all images to 224x224 (or 299x299 for Inception)
4. **Balance classes** — compute class weights + augmentation for underrepresented breeds
5. **Create 10% subset** for local WSL testing
6. **Verify pipeline** — load batch, visualize samples, confirm labels

## Before Any Commit

```bash
# Verify code runs
cd /mnt/c/2025AIDEV/L41
source .venv/bin/activate
python -c "from src.models import get_model; print('Models OK')"
python -c "from src.data.dataset import get_dataloaders; print('Data OK')"
```

## Quick Commands

```bash
# WSL Setup
cd /mnt/c/2025AIDEV/L41
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Run training (10% subset, WSL)
python -m src.training.trainer --model resnet50 --data-fraction 0.1 --epochs 3

# Run evaluation
python -m src.training.evaluate --model resnet50

# Generate plots
python -m src.visualization.plots
```

---

**Version:** 1.0
**Last Updated:** March 2026
**Project:** Dog Breed Classification (L41)
