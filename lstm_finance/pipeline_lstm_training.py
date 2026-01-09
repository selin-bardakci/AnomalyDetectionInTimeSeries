import numpy as np
import pandas as pd
import yfinance as yf

df = yf.download(
    "ETH-USD",   # TSLA-USD yapabilirsin
    start="2017-01-01",
    end="2021-12-31",
    interval="1d",
    auto_adjust=False,
    progress=False
).reset_index()

df = df[["Date", "Close"]].dropna().reset_index(drop=True)

df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

# --- NEW FEATURES ---
df["abs_return"] = df["log_return"].abs()
df["rolling_std_7"]  = df["log_return"].rolling(7).std()
df["rolling_std_30"] = df["log_return"].rolling(30).std()

# Volatility-normalized return (avoid divide-by-zero)
eps = 1e-8
df["norm_return"] = df["log_return"] / (df["rolling_std_30"] + eps)

# Drop NaNs created by rolling windows
df = df.dropna().reset_index(drop=True)

split_ratio = 0.7
split_idx = int(len(df) * split_ratio)

FEATURES = ["norm_return", "abs_return", "rolling_std_7", "rolling_std_30"]

train_feat = df[FEATURES].values[:split_idx]
test_feat  = df[FEATURES].values[split_idx:]

train_dates = df["Date"].values[:split_idx]
test_dates  = df["Date"].values[split_idx:]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_feat)  # shape: (N, 4)
test_scaled  = scaler.transform(test_feat)       # shape: (M, 4)

import joblib, os

ARTIFACT_DIR = "/content/drive/MyDrive/anomaly_project/artifacts_lstm"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

joblib.dump(
    scaler,
    f"{ARTIFACT_DIR}/lstm_return_scaler.joblib"
)
WINDOW = 32

import json

meta = {
    "window": WINDOW,
    "split_ratio": split_ratio,
    "task": "LSTM log-return next-step forecasting",
    "anomaly_method": "residual + percentile threshold"
}

with open(
    f"{ARTIFACT_DIR}/lstm_return_meta.json",
    "w"
) as f:
    json.dump(meta, f, indent=2)


def make_windows_multifeat(X, w, target_col=0):
    # X: (T, F)
    Xs, ys = [], []
    for i in range(len(X) - w):
        Xs.append(X[i:i+w, :])          # (w, F)
        ys.append(X[i+w, target_col])   # scalar target (next norm_return)
    return np.array(Xs), np.array(ys)


X_train, y_train = make_windows_multifeat(train_scaled, WINDOW, target_col=0)
X_test,  y_test  = make_windows_multifeat(test_scaled,  WINDOW, target_col=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(42)

inputs = Input(shape=(WINDOW, len(FEATURES)))  # (32, 4)
x = LSTM(64)(inputs)      # ⬅️ TEK FARK BU
out = Dense(1)(x)

model = Model(inputs, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse"
)

es = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[es],
    verbose=1
)
model.save(
    f"{ARTIFACT_DIR}/lstm_return_anomaly_w32.keras"
)

# ==========================================
# 1) RESIDUAL HESAPLAMA
# ==========================================
yhat_tr = model.predict(X_train, verbose=0).ravel()
yhat_te = model.predict(X_test,  verbose=0).ravel()

res_tr = np.abs(y_train - yhat_tr)
res_te = np.abs(y_test  - yhat_te)

print("Train residual mean/std:", res_tr.mean(), res_tr.std())
print("Test  residual mean/std:", res_te.mean(), res_te.std())

# ==========================================
# 2) STRESS MASK OLUŞTURMA
# ==========================================
abs_ret_full = df["abs_return"].values
std30_full   = df["rolling_std_30"].values

abs_ret_test = abs_ret_full[split_idx + WINDOW:]
std30_test   = std30_full[split_idx + WINDOW:]

assert len(abs_ret_test) == len(res_te)

q = 90
abs_ret_thr = np.percentile(abs_ret_test, q)
stress_mask = (abs_ret_test > abs_ret_thr).astype(int)

print(f"Stress threshold (|return| {q}th pct):", abs_ret_thr)
print("Stress day rate:", stress_mask.mean())

# ==========================================
# 3) VOLATILITE-NORMALIZED RESIDUAL
# ==========================================
eps = 1e-8

# Test için normalize residual
norm_res_te = res_te / (std30_test + eps)

# Train için normalize residual
# 1) Basit versiyon: sadece res_tr kullan
# norm_res_tr = res_tr

# 2) Doğru versiyon:
std30_train = df["rolling_std_30"].values[:split_idx]
std30_train = std30_train[WINDOW:]   # window kadar kaydır
norm_res_tr = res_tr / (std30_train + eps)

print("Normalized residual test mean/std:", norm_res_te.mean(), norm_res_te.std())

# ==============================
# K-SWEEP (Threshold analizi)
# ==============================

def mad_threshold(residuals, k=2.5):
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med)) + 1e-12
    return med + k * 1.4826 * mad

thr = mad_threshold(norm_res_tr, k=2.5)

# Sadece residual büyük OLAN ve aynı zamanda stres günü OLAN noktaları anomaly say
raw_anomaly = (norm_res_te > thr).astype(int)
anomaly = ((norm_res_te > thr) & (stress_mask == 1)).astype(int)

print("Residual-only anomaly rate:", raw_anomaly.mean())
print("Final (residual+stress) anomaly rate:", anomaly.mean())
print("Threshold:", thr)
print("Raw anomaly count:", raw_anomaly.sum())
print("Final anomaly count:", anomaly.sum())

# -----------------------------
# 2) Test için ilgili feature'ları hizala
# -----------------------------
# split_idx ve WINDOW zaten tanımlı
abs_ret_full = df["abs_return"].values
std30_full   = df["rolling_std_30"].values

# Train/test böl, pencere kaybını telafi et
abs_ret_test = abs_ret_full[split_idx + WINDOW:]
std30_test   = std30_full[split_idx + WINDOW:]

# Kontrol amaçlı
assert len(abs_ret_test) == len(res_te) == len(y_test)

dates_test = test_dates[WINDOW:]

import matplotlib.pyplot as plt

plt.figure(figsize=(14,4))
plt.plot(dates_test, y_test, label="Log-return", alpha=0.7)

idx = np.where(anomaly == 1)[0]
plt.scatter(dates_test[idx], y_test[idx],
            color="red", s=25, label="Anomaly")

plt.title("Log-return Anomaly Detection (LSTM)")
plt.xlabel("Date")
plt.ylabel("Log-return")
plt.legend()
plt.tight_layout()
plt.show()
# =====================================
# 2021 YILINA FILTRE
# =====================================

# Önce datetime'e çevir (zaten öyleyse gerek yok ama safe)
dates_test_dt = pd.to_datetime(dates_test)

mask_2021 = (dates_test_dt >= "2021-01-01") & (dates_test_dt < "2022-01-01")

dates_2021 = dates_test_dt[mask_2021]
price_2021 = price_test[mask_2021]
anomaly_2021 = anomaly[mask_2021]
plt.figure(figsize=(16,4))
plt.plot(dates_2021, price_2021, label="Price")

idx = np.where(anomaly_2021 == 1)[0]
plt.scatter(dates_2021[idx], price_2021[idx], color="red", s=30, label="Anomaly")

plt.title("Price vs Anomaly (LSTM, 2021 Only)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()
# NAIVE: y[t] ≈ y[t-1]
yhat_naive = X_test[:, -1, 0]   # window'ın son return'ü
from sklearn.metrics import mean_absolute_error

mae_naive = mean_absolute_error(y_test, yhat_naive)
mae_lstm  = mean_absolute_error(y_test, yhat_te)

print("NAIVE MAE:", mae_naive)
print("LSTM  MAE:", mae_lstm)
