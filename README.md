# Advanced Time Series Forecasting — Simple Dataset (LSTM + SARIMAX + SHAP)

This repo is a simple, GitHub-safe implementation for the assessment:
- Synthetic dataset (1 target `y`, 2 exogenous features)
- LSTM model for multi-step forecasting (single-step in this simplified version)
- SARIMAX baseline
- Evaluation: RMSE and WAPE
- SHAP explainability (DeepExplainer) with fallback

**Uploaded screenshot used in README**: `/mnt/data/79947dcb-5f49-4c69-b915-99d0c4c3e9b9.png`

## How to run (local)
1. `python -m venv venv` && activate it
2. `pip install -r requirements.txt`
3. Create outputs folder or placeholder: `git` step below will show how to create it in GitHub
4. Run: `python main.py --epochs 5 --save-output`

## GitHub / GITIngest tips
- Create `outputs/placeholder.txt` in repo to keep folder in GitHub.
- Paste your GitHub repo URL into GitIngest. This project uses synthetic data only.
