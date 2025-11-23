"""
main.py

Simple, GitHub-safe implementation:
- Small synthetic dataset (single target + two exogenous features)
- LSTM model (single-step forecasting using a short history window)
- SARIMAX baseline
- RMSE and WAPE metrics
- SHAP explanation (attempts DeepExplainer, safe fallback)
- Saves outputs into outputs/ (create outputs/placeholder.txt in repo)
"""

import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Statsmodels baseline
import statsmodels.api as sm

# SHAP
import shap

# -----------------------------
# Utility: Metrics
# -----------------------------
def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percent Error (WAPE)
    WAPE = sum(|y_true - y_pred|) / sum(|y_true|)
    """
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float('inf')
    return np.sum(np.abs(y_true - y_pred)) / denom

# -----------------------------
# 1) Generate tiny synthetic dataset
# -----------------------------
def generate_simple_dataset(n_hours: int = 24*90) -> pd.DataFrame:
    """
    Generate a tiny synthetic hourly dataset (90 days by default).
    Columns:
        - y : target
        - feat1 : exogenous feature (sine daily)
        - feat2 : exogenous (random walk)
    """
    np.random.seed(0)
    t = np.arange(n_hours)

    # simple daily seasonality
    feat1 = 5.0 * np.sin(2 * np.pi * t / 24.0)

    # random-walk like exogenous
    noise = np.random.normal(scale=0.5, size=n_hours)
    feat2 = np.cumsum(noise) * 0.1 + 2.0

    # target = baseline + feat contributions + small trend + noise
    trend = 0.001 * t
    y = 20.0 + 0.8 * feat1 + 1.2 * feat2 + trend + np.random.normal(0, 0.5, n_hours)

    df = pd.DataFrame({"y": y, "feat1": feat1, "feat2": feat2})
    return df

# -----------------------------
# 2) Windowing for LSTM
# -----------------------------
def create_windows(df_values: np.ndarray, seq_len: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    df_values: numpy array with shape (n_samples, n_features)
    Returns X (n_windows, seq_len, n_features), y (n_windows,)
    """
    X, y = [], []
    n = len(df_values)
    for i in range(n - seq_len):
        X.append(df_values[i:i+seq_len])
        y.append(df_values[i+seq_len, 0])  # target is column 0
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# -----------------------------
# 3) Build LSTM model
# -----------------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='tanh', return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# -----------------------------
# 4) SARIMAX baseline helper
# -----------------------------
def fit_sarimax(series: np.ndarray, exog: np.ndarray):
    # Use a small SARIMAX config (may not always be ideal)
    try:
        model = sm.tsa.SARIMAX(series, exog=exog, order=(1,0,0), seasonal_order=(0,1,1,24),
                               enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        return res
    except Exception as e:
        print("SARIMAX fit error:", e)
        return None

# -----------------------------
# 5) SHAP explainability
# -----------------------------
def compute_shap(model, X_sample):
    """
    Try DeepExplainer on Keras model. If fails, return None.
    """
    try:
        # Background set: small subset
        background = X_sample[:50].astype(np.float32)
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_sample[:10].astype(np.float32))
        return shap_values
    except Exception as e:
        print("SHAP compute failed:", e)
        return None

# -----------------------------
# MAIN: training + baseline + shap + save
# -----------------------------
def main(args):
    os.makedirs("outputs", exist_ok=True)

    # 1. Data
    df = generate_simple_dataset(n_hours=args.hours)
    print("Data shape:", df.shape)

    # 2. Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)  # y, feat1, feat2

    # 3. Windowing
    seq_len = args.seq_len
    X, y = create_windows(scaled, seq_len=seq_len)
    print("Windows:", X.shape, y.shape)

    # 4. Train-test split (80-20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 5. Build & train LSTM
    model = build_lstm_model(input_shape=(seq_len, X.shape[2]))
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=32, verbose=1)

    # Save model and scaler
    model.save("outputs/model.h5")
    joblib.dump(scaler, "outputs/scaler.pkl")

    # 6. Predictions & evaluation (scale-aware)
    y_pred = model.predict(X_test).flatten()
    rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
    wape_val = wape(y_test, y_pred)

    print(f"LSTM - RMSE (scaled): {rmse_val:.6f}, WAPE (scaled): {wape_val:.6f}")

    # For comparability, invert scaling for a sample (to produce RMSE in original units)
    # We'll invert using scaler: note scaler expects full feature vector; we replace y column with preds/trues and inverse transform one row
    def invert_y(y_scaled_values, X_context):
        """
        Invert the scaled y to original units. This is approximate because scaler is fit on all columns.
        y_scaled_values: 1D array (n,)
        X_context: 3D array representing windows for the corresponding rows to build full feature vector for inverse transform.
        We'll use the last row of each window as the exog context for inverse transform.
        Returns 1D array in original units.
        """
        originals = []
        for i, val in enumerate(y_scaled_values):
            # pick last step of the corresponding window in unscaled space for exog placeholders
            last_window = X_context[i, -1, :].copy()
            arr = last_window.copy()
            arr[0] = val  # set target column to our predicted/true scaled value
            orig = scaler.inverse_transform(arr.reshape(1, -1))[0, 0]
            originals.append(orig)
        return np.array(originals)

    # Prepare context windows aligned with X_test (last seq_len windows correspond to windows in X_test)
    # To invert, we need the last-step exog from each window: use X_test[:, -1, :]
    y_test_orig = invert_y(y_test, X_test)
    y_pred_orig = invert_y(y_pred, X_test)

    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    wape_orig = wape(y_test_orig, y_pred_orig)
    print(f"LSTM - RMSE (orig units): {rmse_orig:.6f}, WAPE (orig units): {wape_orig:.6f}")

    # 7. SARIMAX baseline
    # To train SARIMAX we need a 1D series aligned to same indexing as windows.
    # We'll use original df to align: windows start at index 0 and predict index seq_len .. end
    raw = df.values
    series_for_windows = raw[:, 0]
    # training indices for SARIMAX correspond to seq_len .. seq_len + len(X_train)-1 inclusive
    sarimax_train_series = series_for_windows[seq_len: seq_len + len(X_train)]
    # exog training uses feat1 & feat2 aligned to those indices
    sarimax_exog = raw[seq_len: seq_len + len(X_train), 1:]

    sarimax_res = fit_sarimax(sarimax_train_series, sarimax_exog)
    if sarimax_res is not None:
        exog_forecast = raw[seq_len + len(X_train): seq_len + len(X_train) + len(X_test), 1:]
        try:
            sarimax_pred = sarimax_res.forecast(steps=len(X_test), exog=exog_forecast)
            # ground truth for same horizon:
            ground_truth = series_for_windows[seq_len + len(X_train): seq_len + len(X_train) + len(X_test)]
            sar_rmse = np.sqrt(mean_squared_error(ground_truth, sarimax_pred))
            sar_wape = wape(ground_truth, sarimax_pred)
            print(f"SARIMAX - RMSE (orig): {sar_rmse:.6f}, WAPE (orig): {sar_wape:.6f}")
        except Exception as e:
            print("SARIMAX forecast error:", e)
    else:
        print("SARIMAX model not available; skipping baseline comparison.")

    # 8. SHAP explanation (best-effort)
    print("Attempting SHAP explanation (may be slow on CPU)...")
    shap_vals = compute_shap(model, X_test)
    if shap_vals is not None:
        try:
            # shap_values for Keras DeepExplainer returns list; shap_vals[0] shape ~ (n_samples, seq_len, n_features)
            # We'll aggregate along time to see feature importance by summing absolute values over time.
            arr = np.array(shap_vals[0])  # shape (n_samples, seq_len, n_features)
            # compute mean(|shap|) across sample and time for each feature
            mean_abs = np.mean(np.abs(arr), axis=(0,1))
            feature_names = ["y", "feat1", "feat2"]
            print("SHAP mean absolute contribution (summed over time):")
            for name, val in zip(feature_names, mean_abs):
                print(f"  {name}: {val:.6f}")

            # Save a small visualization
            import matplotlib.pyplot as plt
            plt.bar(feature_names, mean_abs)
            plt.title("Mean absolute SHAP contributions (time-aggregated)")
            plt.tight_layout()
            plt.savefig("outputs/shap_simple_bar.png")
            plt.close()
            print("SHAP bar chart saved to outputs/shap_simple_bar.png")
        except Exception as e:
            print("SHAP post-processing error:", e)
    else:
        print("SHAP not available in this environment or failed. This is non-fatal.")

    # 9. Save a small CSV of predictions (scaled & original)
    out_df = pd.DataFrame({
        "y_test_scaled": y_test.flatten(),
        "y_pred_scaled": y_pred.flatten(),
        "y_test_orig": y_test_orig.flatten(),
        "y_pred_orig": y_pred_orig.flatten()
    })
    out_df.to_csv("outputs/predictions_sample.csv", index=False)
    print("Saved outputs/predictions_sample.csv and model/scaler to outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24*90, help="How many hourly points to generate (default 90 days)")
    parser.add_argument("--seq_len", type=int, default=24, help="Window length (hours)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--save-output", action="store_true", help="Save outputs into outputs/")
    args = parser.parse_args()
    main(args)
