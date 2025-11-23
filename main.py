"""
# Build and train LSTM
model = build_lstm((seq_len, X.shape[2]))


model.fit(X_train, y_train, epochs=args.epochs, batch_size=64, verbose=1)


# Save model and scaler
model.save('outputs/model.h5')
joblib.dump(scaler, 'outputs/scaler.pkl')


# Prediction & evaluation (note predictions are on scaled y)
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"LSTM RMSE (scaled units): {rmse:.4f}")


# Baseline SARIMAX: use the last column ordering from scaled (ensure alignment)
train_idx = np.arange(seq_len, seq_len + len(X_train))
# For SARIMAX we need a 1D series (original y) and exog aligned; use raw df values
raw = df.values
sarimax_train_series = raw[seq_len:seq_len + len(X_train), 0]
sarimax_exog = raw[seq_len:seq_len + len(X_train), 1:]


try:
sarimax_res = sarimax_baseline(sarimax_train_series, sarimax_exog)
# Forecast same length as test
exog_forecast = raw[seq_len + len(X_train): seq_len + len(X_train) + len(X_test), 1:]
sarimax_pred = sarimax_res.forecast(steps=len(X_test), exog=exog_forecast)
sarimax_rmse = np.sqrt(mean_squared_error(raw[seq_len + len(X_train): seq_len + len(X_train) + len(X_test), 0], sarimax_pred))
print(f"SARIMAX RMSE (original units): {sarimax_rmse:.4f}")
except Exception as e:
print("SARIMAX baseline failed to fit (statsmodels may raise errors depending on environment):", e)


# SHAP (explain LSTM predictions)
print("Computing SHAP values (this may take a minute)...")
try:
shap_values = compute_shap_for_lstm(model, X_test)
shap.summary_plot(shap_values[0], X_test[:10], show=False)
plt.tight_layout()
plt.savefig('outputs/shap_summary.png')
plt.close()
print('SHAP summary saved to outputs/shap_summary.png')
except Exception as e:
print('SHAP explanation failed:', e)


if args.save_output:
# Save a small sample of predictions and ground truth for inspection
sample_df = pd.DataFrame({
'y_true_scaled': y_test.flatten()[:200],
'y_pred_scaled': pred.flatten()[:200]
})
sample_df.to_csv('outputs/predictions_sample.csv', index=False)
print('Sample predictions saved to outputs/predictions_sample.csv')




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--save-output', action='store_true', help='Save outputs to outputs/')
args = parser.parse_args()
main(args)
