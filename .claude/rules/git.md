# Git Workflow

## Branch Naming
```
feature/data-pipeline
feature/model-resnet50
feature/transfer-learning
feature/experiments
feature/readme-v1
bugfix/inception-input-size
```

## Commit Messages
```
feat(data): add dataset download and organization pipeline
feat(models): implement ResNet-50 with transfer learning
feat(training): add 3-stage transfer learning pipeline
feat(viz): generate accuracy comparison plots
fix(data): correct class weight computation
docs(readme): add CNN fundamentals section
test(models): add smoke tests for all architectures
```

## Commit Rules

1. Commit after each completed task or logical unit of work
2. Never commit large data files (images, model weights)
3. Never commit credentials or API keys
4. Always verify `.gitignore` covers: `data/`, `results/models/`, `.venv/`, `__pycache__/`
5. Do NOT amend published commits
6. Do NOT force-push to main

## What to Commit

- All `src/` code
- `docs/PRD.md`, `docs/tasks.json`
- `requirements.txt`, `pyproject.toml`
- `.gitignore`
- `README.md`
- `results/graphs/` (PNG files, reasonable size)
- `results/tables/` (CSV files)
- `notebooks/` (Colab notebook)

## What NOT to Commit

- `data/raw/` (Kaggle dataset — too large)
- `data/processed/` (generated from raw)
- `results/models/*.pth` (model weights — too large)
- `.venv/`
- `__pycache__/`
- `.env`, `kaggle.json`, any credentials
