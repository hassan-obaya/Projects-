pip install yfinance pandas ta scikit-learn matplotlib

import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import yfinance as yf
import ta  # technical analysis indicators

import sklearn.preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

import xgboost as xgb

# Download daily price data for AAPL from 2015 to end of 2023
df = yf.download('AAPL', start='2015-01-01', end='2024-01-01')

# Keep only the core OHLCV features
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Simple Moving Average (10-day)
df['MA_10'] = df['Close'].rolling(window=10, min_periods=10).mean()

# Relative Strength Index (14-day)
#    RSIIndicator returns a pandas Series; .rsi() fetches the indicator values
rsi_indicator = ta.momentum.RSIIndicator(close=df['Close'].squeeze(), window=14)
df['RSI_14'] = rsi_indicator.rsi()

# Drop initial rows that contain NaNs due to indicator windows
df.dropna(inplace=True)

print("Data snapshot:")
print(df.head())
print("\nShape:", df.shape)

plt.figure(figsize=(15, 5))

# Price plot
plt.subplot(1, 2, 1)
plt.plot(df.index, df['Open'],  label='Open',  linestyle='--')
plt.plot(df.index, df['High'],  label='High',   alpha=0.6)
plt.plot(df.index, df['Low'],   label='Low',    alpha=0.6)
plt.plot(df.index, df['Close'], label='Close',  linewidth=2)
plt.title('AAPL Price Evolution (2015–2023)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()

# Volume plot
plt.subplot(1, 2, 2)
plt.plot(df.index, df['Volume'], label='Volume')
plt.title('AAPL Trading Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()

plt.tight_layout()
plt.show()

# Select features for the model
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'RSI_14']
data = df[FEATURE_COLUMNS].values  # as NumPy array

# Scale features to [0, 1] using Min-Max scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(
    data: np.ndarray,
    seq_len: int = 60,
    target_col_idx: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the time series data into sequences for LSTM:
      - X: sequences of length `seq_len` (each sequence is seq_len × num_features)
      - y: next-day target (the 'Close' price at index seq_len + i)
    
    Args:
        data           : numpy array of shape (n_samples, n_features)
        seq_len        : length of the input sequence (default 60 days)
        target_col_idx : which column to predict (default index 3 → 'Close')
    
    Returns:
        X : np.ndarray of shape (n_samples−seq_len−1, seq_len, n_features)
        y : np.ndarray of shape (n_samples−seq_len−1,)
    """
    X, y = [], []
    for i in range(seq_len, len(data) - 1):
        X.append(data[i - seq_len:i, :])
        # The label is the day AFTER the end of the sequence
        y.append(data[i + 1, target_col_idx])
    return np.array(X), np.array(y)

# Generate sequences (60-day windows predicting next day's Close)
SEQ_LEN = 60
X, y = create_sequences(scaled_data, seq_len=SEQ_LEN)

n = len(X)
train_end = int(0.70 * n)
val_end   = train_end + int(0.15 * n)

X_train, y_train = X[:train_end], y[:train_end]
X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
X_test,  y_test  = X[val_end:], y[val_end:]

print(f"Dataset sizes → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

model = Sequential([
    # First LSTM layer returns sequences so we can stack another LSTM
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    # Second LSTM layer outputs a vector of size 64
    LSTM(64),
    Dropout(0.2),
    # Final dense layer predicts a single value (next-day Close price)
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
model.summary()

EPOCHS = 50
BATCH_SIZE = 32

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)

def invert_scale(scaled_vals: np.ndarray, col_idx: int = 3) -> np.ndarray:
    """
    Inverse-transform the scaled values for the given target column.
    We build a dummy matrix to match the scaler's expected input shape.
    """
    # Create dummy array: zeros for all other features, insert our column of interest
    dummy = np.zeros((len(scaled_vals), data.shape[1]))
    dummy[:, col_idx] = scaled_vals.ravel()
    # Inverse transform entire array, then extract target column
    inv = scaler.inverse_transform(dummy)
    return inv[:, col_idx]

# Predict on each split
pred_train = model.predict(X_train)
pred_val   = model.predict(X_val)
pred_test  = model.predict(X_test)

# Convert scaled predictions & true labels back to USD
y_train_real = invert_scale(y_train)
y_val_real   = invert_scale(y_val)
y_test_real  = invert_scale(y_test)

pred_train_real = invert_scale(pred_train)
pred_val_real   = invert_scale(pred_val)
pred_test_real  = invert_scale(pred_test)

pred_train = model.predict(X_train)
pred_val = model.predict(X_val)
pred_test = model.predict(X_test)


all_targets_real = np.concatenate([y_train_real, y_val_real, y_test_real])
all_preds_real = np.concatenate([pred_train_real, pred_val_real, pred_test_real])

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(len(y_train_real)), y_train_real, label="train target", color='blue')
plt.plot(range(len(y_train_real), len(y_train_real) + len(y_val_real)), y_val_real, label="valid target", color='gray')
plt.plot(range(len(y_train_real) + len(y_val_real), len(all_targets_real)), y_test_real, label="test target", color='black')

plt.plot(range(len(y_train_real)), pred_train_real, label="train prediction", color='red')
plt.plot(range(len(y_train_real), len(y_train_real) + len(y_val_real)), pred_val_real, label="valid prediction", color='orange')
plt.plot(range(len(y_train_real) + len(y_val_real), len(all_targets_real)), pred_test_real, label="test prediction", color='green')

plt.title('Stock Prices (USD)')
plt.xlabel('Time [days]')
plt.ylabel('Close Price (USD)')
plt.legend()


plt.subplot(1, 2, 2)
start = len(y_train_real) + len(y_val_real)
end = len(all_targets_real)
plt.plot(range(start, end), y_test_real, label="test target", color='black')
plt.plot(range(start, end), pred_test_real, label="test prediction", color='green')
plt.title('Test Forecast (USD)')
plt.xlabel('Time [days]')
plt.ylabel('Close Price (USD)')
plt.legend()

plt.tight_layout()
plt.show()

# =============================================================================
# 14. Evaluate Performance with MAE and RMSE
# =============================================================================
def report_metrics(true_vals, pred_vals, label=""):
    mae  = mean_absolute_error(true_vals, pred_vals)
    rmse = math.sqrt(mean_squared_error(true_vals, pred_vals))
    print(f"{label:10s} MAE: {mae:8.2f} | RMSE: {rmse:8.2f}")

print("\nPerformance Metrics (USD):")
report_metrics(y_train_real, pred_train_real, label="Train")
report_metrics(y_val_real,   pred_val_real,   label="Val")
report_metrics(y_test_real,  pred_test_real,  label="Test")