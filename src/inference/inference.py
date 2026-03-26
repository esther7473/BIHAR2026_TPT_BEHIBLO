import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from src.common.common import CONFIG, ROOT_DIR
from src.data.database import load_weather_data, save_predictions
from src.training.preprocessing import run_preprocessing, create_sequences, prepare_inference_input, FEATURE_COLS
from src.training.train import inverse_transform_multi
import mlflow
import pickle


def run_inference(run_date=None):

    lookback   = CONFIG["model"]["lookback"]
    horizon    = CONFIG["model"]["horizon"]
    model_name = CONFIG["model"]["name"]

    # ── MLflow ──
    mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
    model = mlflow.keras.load_model(f"models:/{model_name}@champion")
    print(f" Modèle chargé : models:/{model_name}@champion")

    # ── Données ──
    df = load_weather_data()
    if run_date is not None:
        df = df[df.index <= run_date]

    # ── Scaler ──
    scaler_path = CONFIG["paths"]["scaler_path"] 
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # ── Preprocessing minimal ──
    df_resampled, scaled_data, last_sequence = prepare_inference_input(df, scaler, lookback)

    # ── Prédiction ──
    n_features      = scaled_data.shape[1]
    X_input         = last_sequence.reshape(1, lookback, n_features)
    forecast_scaled = model.predict(X_input)[0]  

    # ── Inverse scaling ──
    forecast_real = inverse_transform_multi(
        forecast_scaled.reshape(1, -1),
        scaler,
        n_features
    )[0]  # (horizon,)

    # ── Timestamps futurs ──
    forecast_index = pd.date_range(
        start=df_resampled.index[-1] + pd.Timedelta(hours=3),
        periods=horizon,
        freq="3h"   
    )

    # ── Sauvegarde ──
    save_predictions(
        model_name=model_name,
        timestamps=forecast_index,
        predictions=forecast_real,
        horizon=horizon
    )

    print(f" Prédictions sauvegardées ({forecast_index[0]} - {forecast_index[-1]})")

    return forecast_index, forecast_real


if __name__ == "__main__":
    forecast_index, forecast_real = run_inference()

    # # Affichage rapide
    # df_forecast = pd.DataFrame({
    #     "timestamp":   forecast_index,
    #     "temperature": forecast_real
    # })
    # print(df_forecast)