import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# ✅ mock AVANT l'import de inference
sys.modules["tensorflow"]                   = MagicMock()
sys.modules["tensorflow.keras"]             = MagicMock()
sys.modules["tensorflow.keras.models"]      = MagicMock()
sys.modules["tensorflow.keras.layers"]      = MagicMock()
sys.modules["tensorflow.keras.callbacks"]   = MagicMock()
sys.modules["tensorflow.keras.optimizers"]  = MagicMock()
sys.modules["mlflow"]                       = MagicMock()
sys.modules["mlflow.keras"]                 = MagicMock()

from src.inference.inference import run_inference


# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.random.rand(1, 24).astype("float32")
    return model


@pytest.fixture
def mock_scaler():
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(np.random.uniform(0, 30, (1000, 5)))
    return scaler


@pytest.fixture
def mock_df():
    idx = pd.date_range("2025-01-01", periods=500, freq="h")
    return pd.DataFrame({"temperature": np.random.uniform(0, 20, 500)}, index=idx)


# ─────────────────────────────────────────
# Tests
# ─────────────────────────────────────────

def test_run_inference_returns_correct_shapes(mock_model, mock_scaler, mock_df):
    with patch("src.inference.inference.mlflow.keras.load_model", return_value=mock_model), \
         patch("src.inference.inference.load_weather_data",        return_value=mock_df), \
         patch("src.inference.inference.pickle.load",              return_value=mock_scaler), \
         patch("src.inference.inference.save_predictions"), \
         patch("builtins.open", MagicMock()), \
         patch("src.inference.inference.mlflow.set_tracking_uri"):

        forecast_index, forecast_real = run_inference()

        assert len(forecast_index) == 24
        assert len(forecast_real)  == 24


def test_run_inference_timestamps_in_future(mock_model, mock_scaler, mock_df):
    with patch("src.inference.inference.mlflow.keras.load_model", return_value=mock_model), \
         patch("src.inference.inference.load_weather_data",        return_value=mock_df), \
         patch("src.inference.inference.pickle.load",              return_value=mock_scaler), \
         patch("src.inference.inference.save_predictions"), \
         patch("builtins.open", MagicMock()), \
         patch("src.inference.inference.mlflow.set_tracking_uri"):

        forecast_index, _ = run_inference()

        diffs = forecast_index[1:] - forecast_index[:-1]
        assert all(d == pd.Timedelta(hours=3) for d in diffs)


def test_run_inference_with_run_date(mock_model, mock_scaler, mock_df):
    with patch("src.inference.inference.mlflow.keras.load_model", return_value=mock_model), \
         patch("src.inference.inference.load_weather_data",        return_value=mock_df), \
         patch("src.inference.inference.pickle.load",              return_value=mock_scaler), \
         patch("src.inference.inference.save_predictions"), \
         patch("builtins.open", MagicMock()), \
         patch("src.inference.inference.mlflow.set_tracking_uri"):

        forecast_index, forecast_real = run_inference(run_date="2025-01-10")

        assert len(forecast_index) == 24
        assert len(forecast_real)  == 24


def test_run_inference_calls_save_predictions(mock_model, mock_scaler, mock_df):
    with patch("src.inference.inference.mlflow.keras.load_model", return_value=mock_model), \
         patch("src.inference.inference.load_weather_data",        return_value=mock_df), \
         patch("src.inference.inference.pickle.load",              return_value=mock_scaler), \
         patch("src.inference.inference.save_predictions") as mock_save, \
         patch("builtins.open", MagicMock()), \
         patch("src.inference.inference.mlflow.set_tracking_uri"):

        run_inference()

        mock_save.assert_called_once()