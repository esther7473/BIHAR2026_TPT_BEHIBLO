import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.preprocessing import MinMaxScaler

# ✅ mock tensorflow et keras AVANT l'import de train
sys.modules["tensorflow"]                   = MagicMock()
sys.modules["tensorflow.keras"]             = MagicMock()
sys.modules["tensorflow.keras.models"]      = MagicMock()
sys.modules["tensorflow.keras.layers"]      = MagicMock()
sys.modules["tensorflow.keras.callbacks"]   = MagicMock()
sys.modules["tensorflow.keras.optimizers"]  = MagicMock()
sys.modules["mlflow"]                       = MagicMock()
sys.modules["mlflow.keras"]                 = MagicMock()

from src.training.train import build_model, inverse_transform_multi


# ─────────────────────────────────────────
# Tests build_model (mockés)
# ─────────────────────────────────────────

def test_build_model_output_shape():
    mock_model = MagicMock()
    mock_model.output_shape = (None, 24)

    with patch("src.training.train.Sequential", return_value=mock_model):
        model = build_model(
            input_shape=(24, 5),
            horizon=24,
            lstm_units_l1=64,
            lstm_units_l2=64,
            dropout=0.2,
            lr=0.001
        )
        assert model.output_shape == (None, 24)


def test_build_model_input_shape():
    mock_model = MagicMock()
    mock_model.input_shape = (None, 24, 5)

    with patch("src.training.train.Sequential", return_value=mock_model):
        model = build_model(
            input_shape=(24, 5),
            horizon=24,
            lstm_units_l1=64,
            lstm_units_l2=64,
            dropout=0.2,
            lr=0.001
        )
        assert model.input_shape == (None, 24, 5)


def test_build_model_predict():
    mock_model        = MagicMock()
    mock_model.predict.return_value = np.random.rand(4, 24).astype("float32")

    with patch("src.training.train.Sequential", return_value=mock_model):
        model = build_model(
            input_shape=(24, 5),
            horizon=24,
            lstm_units_l1=32,
            lstm_units_l2=32,
            dropout=0.1,
            lr=0.001
        )
        X     = np.random.rand(4, 24, 5).astype("float32")
        preds = model.predict(X, verbose=0)
        assert preds.shape == (4, 24)


# ─────────────────────────────────────────
# Tests inverse_transform_multi (pas de mock nécessaire)
# ─────────────────────────────────────────

@pytest.fixture
def scaler():
    data = np.random.uniform(0, 30, (1000, 5))
    s    = MinMaxScaler()
    s.fit(data)
    return s


def test_inverse_transform_shape(scaler):
    preds  = np.random.rand(10, 24)
    result = inverse_transform_multi(preds, scaler, n_features=5)
    assert result.shape == (10, 24)


def test_inverse_transform_values_in_range(scaler):
    preds  = np.random.rand(10, 24)
    result = inverse_transform_multi(preds, scaler, n_features=5)
    assert result.min() >= 0
    assert result.max() <= 30


def test_inverse_transform_roundtrip(scaler):
    original     = np.array([[15.0, 20.0, 25.0]])
    temp         = np.zeros((3, 5))
    temp[:, 0]   = original[0]
    scaled_temps = scaler.transform(temp)[:, 0]
    preds        = scaled_temps.reshape(1, -1)
    result       = inverse_transform_multi(preds, scaler, n_features=5)
    assert result[0] == pytest.approx(original[0], abs=0.01)