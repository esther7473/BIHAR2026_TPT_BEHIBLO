import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.training.preprocessing import (
    resample_3h, add_cyclic_features, remove_missing,
    train_test_split_ts, scale_data, create_sequences,
    run_preprocessing, prepare_inference_input, FEATURE_COLS
)

@pytest.fixture
def df_hourly():
    """DataFrame horaire avec 3000 points pour les tests."""
    idx = pd.date_range("2020-01-01", periods=3000, freq="h")
    return pd.DataFrame({"temperature": np.random.uniform(0, 20, 3000)}, index=idx)



def test_resample_3h(df_hourly):
    df = resample_3h(df_hourly)

    assert "temperature" in df.columns
    assert len(df) == len(df_hourly) // 3
    expected_mean = df_hourly["temperature"].iloc[:3].mean()
    assert df["temperature"].iloc[0] == pytest.approx(expected_mean)


def test_add_cyclic_features(df_hourly):
    df = resample_3h(df_hourly)
    df = add_cyclic_features(df)

    assert list(df.columns) == FEATURE_COLS
    assert df["hour_sin"].between(-1, 1).all()
    assert df["hour_cos"].between(-1, 1).all()
    assert df["month_sin"].between(-1, 1).all()
    assert df["month_cos"].between(-1, 1).all()


def test_remove_missing():
    idx = pd.date_range("2020-01-01", periods=5, freq="3h")
    df  = pd.DataFrame({"temperature": [1.0, None, 3.0, None, 5.0]}, index=idx)
    df  = remove_missing(df)
    assert len(df) == 3
    assert df.isna().sum().sum() == 0


def test_train_test_split_ts(df_hourly):
    df = resample_3h(df_hourly)
    df = add_cyclic_features(df)

    with patch("src.training.preprocessing.CONFIG",
               {"data": {"test_size": 100}, "model": {"lookback": 24, "horizon": 24}}):
        train, test = train_test_split_ts(df)
        assert len(test)  == 100
        assert len(train) == len(df) - 100
        assert train.index.max() < test.index.min()


def test_scale_data(df_hourly):
    df    = resample_3h(df_hourly)
    df    = add_cyclic_features(df)
    train = df.iloc[:-100]
    test  = df.iloc[-100:]

    scaled_train, scaled_test, scaler = scale_data(train, test)

    assert scaled_train.shape == (len(train), 5)
    assert scaled_test.shape  == (len(test),  5)

    # ✅ train est entre 0 et 1 car scaler fitté dessus
    assert scaled_train.min() >= 0
    assert scaled_train.max() <= 1

    # ✅ test peut être hors [0,1] — c'est normal
    assert scaled_test.shape == (len(test), 5)  # juste vérifier la shape


def test_prepare_inference_input(df_hourly):
    with patch("src.training.preprocessing.CONFIG", {
        "data":  {"resample_freq": "3h", "test_size": 100},
        "model": {"lookback": 24, "horizon": 24}
    }):
        df           = resample_3h(df_hourly)
        df           = add_cyclic_features(df)
        _, _, scaler = scale_data(df.iloc[:-100], df.iloc[-100:])

        df_resampled, scaled, last_seq = prepare_inference_input(df_hourly, scaler, lookback=24)

        assert last_seq.shape == (24, 5)
        assert scaled.shape[1] == 5          # ✅ 5 features
        assert df_resampled.index.name == "timestamp"



def test_create_sequences():
    data    = np.random.rand(200, 5)
    lookback = 24
    horizon  = 24

    X, y = create_sequences(data, lookback, horizon)

    assert X.shape == (200 - lookback - horizon, lookback, 5)
    assert y.shape == (200 - lookback - horizon, horizon)
    assert y.ndim  == 2


def test_run_preprocessing(df_hourly):
    with patch("src.training.preprocessing.CONFIG", {
        "data":  {"resample_freq": "3h", "test_size": 100},
        "model": {"lookback": 24, "horizon": 24}
    }):
        data = run_preprocessing(df_hourly)

        assert "X_train" in data
        assert "y_train" in data
        assert "X_test"  in data
        assert "y_test"  in data
        assert "scaler"  in data

        assert data["X_train"].ndim == 3
        assert data["y_train"].ndim == 2
        assert data["X_train"].shape[1] == 24  # lookback
        assert data["y_train"].shape[1] == 24  # horizon

