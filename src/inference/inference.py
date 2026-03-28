import numpy as np
import pandas as pd
import pickle
import keras

from src.common.common import CONFIG
from src.data.database import load_weather_data, save_predictions
from src.training.preprocessing import prepare_inference_input, FEATURE_COLS
from src.training.train import inverse_transform_multi


def run_inference(run_date=None):

    lookback   = CONFIG["model"]["lookback"]
    horizon    = CONFIG["model"]["horizon"]
    model_name = CONFIG["model"]["name"]

    model_path = CONFIG["paths"]["model_path"]
    model      = keras.models.load_model(model_path)
    print(f" Modèle chargé : {model_path}")

    df = load_weather_data()
    if run_date is not None:
        df = df[df.index <= run_date]

    if len(df) < lookback:
        raise ValueError(f"Pas assez de données : {len(df)} points, {lookback} requis.")

    scaler_path = CONFIG["paths"]["scaler_path"]
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    df_resampled, scaled_data, last_sequence = prepare_inference_input(df, scaler, lookback)

    n_features      = scaled_data.shape[1]
    X_input         = last_sequence.reshape(1, lookback, n_features)
    forecast_scaled = model.predict(X_input)[0]

    forecast_real = inverse_transform_multi(
        forecast_scaled.reshape(1, -1),
        scaler,
        n_features
    )[0]  

    # ── Timestamps futurs ──
    forecast_index = pd.date_range(
        start=df_resampled.index[-1] + pd.Timedelta(hours=3),
        periods=horizon,
        freq="3h"
    )

    horizons = list(range(1, horizon + 1))  
    save_predictions(
        model_name=model_name,
        timestamps=forecast_index,
        predictions=forecast_real,
        horizon=horizons
    )

    print(f" Prédictions sauvegardées ({forecast_index[0]} → {forecast_index[-1]})")
    return forecast_index, forecast_real


if __name__ == "__main__":
    forecast_index, forecast_real = run_inference()