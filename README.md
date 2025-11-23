# AI_project1
# Advanced Time Series Forecasting with Deep Learning and Explainability


This repository implements a synthetic multivariate time-series forecasting project required for the assessment. It generates a multi-year hourly dataset (trend, multiple seasonalities, exogenous variables), trains an LSTM, evaluates against a SARIMAX baseline, and computes SHAP explanations.


**Important:** The dataset is generated programmatically (no real/personal data). This repository is safe to upload to GitHub and use with GITIngest.


### How to run


1. Create a Python virtual environment and install requirements:


```bash
python -m venv venv
source venv/bin/activate # macOS / Linux
venv\Scripts\activate # Windows (PowerShell)
pip install -r requirements.txt
