import pandas as pd
# from database import save_weather_data

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.data.database import save_weather_data
from src.common.common import CONFIG

FEATURE_COLS = ["temperature", "hour_sin", "hour_cos", "month_sin", "month_cos"]


# ─────────────────────────────────────────
# Étapes individuelles
# ─────────────────────────────────────────

def clean_raw(df):
    """Supprime les colonnes inutiles et renomme latitude/longitude → timestamp/temperature."""
    df = df.drop(columns=["elevation", "utc_offset_seconds", "timezone", "timezone_abbreviation"])
    df = df.drop(df.index[[0, 1]])
    df = df.rename(columns={"latitude": "timestamp", "longitude": "temperature"})
    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df["temperature"] = pd.to_numeric(df["temperature"])
    df = df.set_index("timestamp")
    return df


def resample_3h(df):
    """Rééchantillonne la température à 3h et retourne un DataFrame."""
    freq = CONFIG["data"]["resample_freq"]   # ex: "3h"
    return df["temperature"].resample(freq).mean().to_frame()


def add_cyclic_features(df):
    """Ajoute les features cycliques heure et mois (sin/cos)."""
    df = df.copy()
    df["hour_sin"]   = np.sin(2 * np.pi * df.index.hour  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df.index.hour  / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df.index.month / 12)
    return df[FEATURE_COLS]


def remove_missing(df):
    return df.dropna()


def train_test_split_ts(df):
    """Split temporel : 1 an de test (8 points/jour × 365 jours)."""
    test_size = CONFIG["data"]["test_size"]   
    train = df.iloc[:-test_size]
    test  = df.iloc[-test_size:]
    return train, test


def scale_data(train, test):
    """Fit le scaler sur train, transforme train et test."""
    scaler       = MinMaxScaler()
    scaled_train = scaler.fit_transform(train)
    scaled_test  = scaler.transform(test)
    return scaled_train, scaled_test, scaler


def create_sequences(data, lookback, horizon):
    """
    Construit les séquences (X, y) pour le LSTM multioutput.
    X : (N, lookback, n_features)
    y : (N, horizon)  — température uniquement (col 0)
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon, 0])

    X, y = np.array(X), np.array(y)
    assert y.ndim == 2, f"y devrait être 2D, got {y.shape}"
    return X, y


# ─────────────────────────────────────────
# Pipeline principale
# ─────────────────────────────────────────

def run_preprocessing(df_raw):
    """
    Entrée  : df issu de load_weather_data() — index datetime, colonne temperature
    Sortie  : dict avec X_train, y_train, X_test, y_test, scaler, train, test, df
    """
    lookback = CONFIG["model"]["lookback"]
    horizon  = CONFIG["model"]["horizon"]

    # 1. Resample + features cycliques
    df = resample_3h(df_raw)        # → (N, 1) colonne temperature à 3h
    df = add_cyclic_features(df)    # → (N, 5) + hour_sin/cos, month_sin/cos
    df = remove_missing(df)

    # 2. Split temporel
    train, test = train_test_split_ts(df)

    # 3. Normalisation
    scaled_train, scaled_test, scaler = scale_data(train, test)

    # 4. Séquences LSTM
    X_train, y_train = create_sequences(scaled_train, lookback, horizon)
    X_test,  y_test  = create_sequences(scaled_test,  lookback, horizon)

    print(f"Total  : {len(df)} points")
    print(f"Train  : {len(train)} | X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"Test   : {len(test)}  | X_test : {X_test.shape}  | y_test : {y_test.shape}")

    return {
        "df":      df,
        "train":   train,
        "test":    test,
        "scaler":  scaler,
        "scaled_data": np.vstack([scaled_train, scaled_test]),  # ← toutes les données normalisées
        "X_train": X_train,
        "y_train": y_train,
        "X_test":  X_test,
        "y_test":  y_test,
    }

def prepare_inference_input(df_raw, scaler, lookback):
    df     = resample_3h(df_raw)
    df     = add_cyclic_features(df)
    df     = remove_missing(df)
    scaled = scaler.transform(df)          # ← passe le DataFrame directement
    last_sequence = scaled[-lookback:]
    return df, scaled, last_sequence