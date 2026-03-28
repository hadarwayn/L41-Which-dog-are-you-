# Security & Data Safety

## Credentials
- Never commit Kaggle API keys (`kaggle.json`)
- Never hardcode API keys in source code
- Use environment variables or `.env` files (gitignored)
- Never commit Google Drive tokens

## Data Safety
- Always verify `.gitignore` before committing
- Never commit image datasets (too large, may have license restrictions)
- Model weights (`.pth` files) stay local or on Google Drive — not in git
- Celebrity images are for educational/fair-use purposes only

## Before Every Commit Checklist
- [ ] No API keys or tokens in code
- [ ] No large binary files staged
- [ ] `.gitignore` is up to date
- [ ] No `data/` or `*.pth` files in staging area
