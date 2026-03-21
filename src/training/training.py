import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from common import CONFIG


# ─────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────

def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]


def compute_metrics(y_true, y_pred):
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae":  mean_absolute_error(y_true, y_pred)
    }


# ─────────────────────────────────────────
# ARIMA
# ─────────────────────────────────────────

def train_arima(train, tscv):
    order       = tuple(CONFIG["arima"]["order"])
    rmse_scores = []
    mae_scores  = []

    for train_idx, val_idx in tscv.split(train):
        train_fold = train.iloc[train_idx]
        val_fold   = train.iloc[val_idx]

        model    = SARIMAX(train_fold, order=order).fit(disp=False)
        forecast = model.forecast(steps=len(val_fold))
        metrics  = compute_metrics(val_fold, forecast)

        rmse_scores.append(metrics["rmse"])
        mae_scores.append(metrics["mae"])

    # Modèle final sur tout le train
    final_model = SARIMAX(train, order=order).fit(disp=False)

    print(f"ARIMA  → RMSE CV : {np.mean(rmse_scores):.3f} | MAE CV : {np.mean(mae_scores):.3f}")

    return {
        "model":       final_model,
        "rmse_cv":     np.mean(rmse_scores),
        "mae_cv":      np.mean(mae_scores),
    }


# ─────────────────────────────────────────
# SARIMA
# ─────────────────────────────────────────

def train_sarima(train, tscv):
    order          = tuple(CONFIG["sarima"]["order"])
    seasonal_order = tuple(CONFIG["sarima"]["seasonal_order"])
    rmse_scores    = []
    mae_scores     = []

    for train_idx, val_idx in tscv.split(train):
        train_fold = train.iloc[train_idx]
        val_fold   = train.iloc[val_idx]

        model    = SARIMAX(train_fold, order=order, seasonal_order=seasonal_order).fit(disp=False)
        forecast = model.forecast(steps=len(val_fold))
        metrics  = compute_metrics(val_fold, forecast)

        rmse_scores.append(metrics["rmse"])
        mae_scores.append(metrics["mae"])

    final_model = SARIMAX(train, order=order, seasonal_order=seasonal_order).fit(disp=False)

    print(f"SARIMA → RMSE CV : {np.mean(rmse_scores):.3f} | MAE CV : {np.mean(mae_scores):.3f}")

    return {
        "model":   final_model,
        "rmse_cv": np.mean(rmse_scores),
        "mae_cv":  np.mean(mae_scores),
    }


# ─────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────

def build_lstm_model(lookback, n_units):
    model = Sequential([
        LSTM(n_units, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(CONFIG["model"]["dropout"]),
        LSTM(n_units // 2),
        Dropout(CONFIG["model"]["dropout"]),
        BatchNormalization(),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=CONFIG["model"]["learning_rate"]),
        loss='mse',
        metrics=['mean_absolute_error']
    )
    return model


def train_lstm(train, tscv):
    lookback    = CONFIG["model"]["lookback"]
    n_units     = CONFIG["model"]["n_units"]
    scaler      = MinMaxScaler()
    scaled_train = scaler.fit_transform(train.values.reshape(-1, 1))

    rmse_scores = []
    mae_scores  = []

    for train_idx, val_idx in tscv.split(scaled_train):
        X_train_fold = scaled_train[train_idx]
        X_val_fold   = scaled_train[val_idx]

        train_gen = TimeseriesGenerator(X_train_fold, X_train_fold, length=lookback, batch_size=CONFIG["model"]["batch_size"])

        # Contexte lookback pour la validation
        context      = scaled_train[train_idx[-lookback:]]
        X_val_ctx    = np.concatenate([context, X_val_fold])
        val_gen      = TimeseriesGenerator(X_val_ctx, X_val_ctx, length=lookback, batch_size=CONFIG["model"]["batch_size"])

        model = build_lstm_model(lookback, n_units)
        model.fit(train_gen, validation_data=val_gen, epochs=CONFIG["model"]["epochs"], callbacks=get_callbacks(), verbose=0)

        val_pred     = model.predict(val_gen)
        val_pred_inv = scaler.inverse_transform(val_pred)
        val_true_inv = scaler.inverse_transform(X_val_fold[lookback:].reshape(-1, 1))

        metrics = compute_metrics(val_true_inv, val_pred_inv)
        rmse_scores.append(metrics["rmse"])
        mae_scores.append(metrics["mae"])

    # Modèle final
    train_gen_final = TimeseriesGenerator(scaled_train, scaled_train, length=lookback, batch_size=CONFIG["model"]["batch_size"])
    final_model     = build_lstm_model(lookback, n_units)
    final_model.fit(train_gen_final, epochs=CONFIG["model"]["epochs"], callbacks=get_callbacks(), verbose=0)

    print(f"LSTM   → RMSE CV : {np.mean(rmse_scores):.3f} | MAE CV : {np.mean(mae_scores):.3f}")

    return {
        "model":   final_model,
        "scaler":  scaler,
        "rmse_cv": np.mean(rmse_scores),
        "mae_cv":  np.mean(mae_scores),
    }