import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from src.common.common import CONFIG
from database import save_predictions, get_last_timestamp, load_weather_data,get_context




# ─────────────────────────────────────────
# PREDICTIONS PAR MODELE
# ─────────────────────────────────────────

def predict_arima(model_result, horizon):
    model    = model_result["model"]
    forecast = model.forecast(steps=horizon)
    return forecast.values


def predict_sarima(model_result, horizon):
    model    = model_result["model"]
    forecast = model.forecast(steps=horizon)
    return forecast.values


def predict_lstm(model_result, horizon):
    """Fenêtre glissante : prédit point par point"""
    model    = model_result["model"]
    scaler   = model_result["scaler"]
    lookback = CONFIG["model"]["lookback"]

    # Récupérer le contexte depuis la BDD
    context     = get_context(n_points=lookback)
    context_sc  = scaler.transform(context.values.reshape(-1, 1))

    predictions = []
    window      = context_sc.copy()

    for _ in range(horizon):
        x    = window.reshape(1, lookback, 1)
        pred = model.predict(x, verbose=0)
        predictions.append(pred[0, 0])

        # Glisser la fenêtre
        window = np.append(window[1:], pred, axis=0)

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()


# ─────────────────────────────────────────
# ORCHESTRATION
# ─────────────────────────────────────────

def run_inference(models, test, inference_date=None):
    horizon        = CONFIG["inference"]["horizon"]
    model_version  = CONFIG["inference"]["model_version"]
    inference_date = inference_date or pd.Timestamp.now().strftime("%Y-%m-%d")

    predict_fn = {
        "ARIMA":  predict_arima,
        "SARIMA": predict_sarima,
        "LSTM":   predict_lstm
    }

    for model_name, model_result in models.items():
        print(f" Inference {model_name}...")

        # Générer les prédictions
        predictions = predict_fn[model_name](model_result, horizon)

        # Timestamps des prédictions
        last_date  = get_last_timestamp()
        freq       = CONFIG["data"]["resample_freq"]
        timestamps = pd.date_range(
            start=last_date + pd.Timedelta(hours=3),
            periods=horizon,
            freq=freq
        )

        # Métriques sur le test
        test_aligned = test.iloc[-len(predictions):]
        rmse = root_mean_squared_error(test_aligned, predictions)
        mae  = mean_absolute_error(test_aligned, predictions)
        print(f"   RMSE test : {rmse:.3f} | MAE test : {mae:.3f}")

        # Sauvegarder en BDD
        save_predictions(
            model_name    = model_name,
            model_version = model_version,
            inference_date= inference_date,
            timestamps    = timestamps,
            predictions   = predictions,
            horizon       = horizon
        )

    print(" Inference terminée")