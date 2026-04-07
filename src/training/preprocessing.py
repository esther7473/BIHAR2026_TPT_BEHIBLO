import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.data.database import save_weather_data
from src.common.common import CONFIG

FEATURE_COLS = ["temperature", "hour_sin", "hour_cos", "month_sin", "month_cos"]



def clean_raw(df):
    df = df.drop(columns=["elevation", "utc_offset_seconds", "timezone", "timezone_abbreviation"])
    df = df.drop(df.index[[0, 1]])
    df = df.rename(columns={"latitude": "timestamp", "longitude": "temperature"})
    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df["temperature"] = pd.to_numeric(df["temperature"])
    df = df.set_index("timestamp")
    return df


def resample_3h(df):
    freq = CONFIG["data"]["resample_freq"]  
    return df["temperature"].resample(freq).mean().to_frame()


def add_cyclic_features(df):
    df = df.copy()
    df["hour_sin"]   = np.sin(2 * np.pi * df.index.hour  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df.index.hour  / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df.index.month / 12)
    return df[FEATURE_COLS]


def remove_missing(df):
    return df.dropna()


def train_test_split_ts(df):
    test_size = CONFIG["data"]["test_size"]   
    train = df.iloc[:-test_size]
    test  = df.iloc[-test_size:]
    return train, test


def scale_data(train, test):
    scaler       = MinMaxScaler()
    scaled_train = scaler.fit_transform(train)
    scaled_test  = scaler.transform(test)
    return scaled_train, scaled_test, scaler


def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon, 0])

    X, y = np.array(X), np.array(y)
    assert y.ndim == 2, f"y devrait être 2D, got {y.shape}"
    return X, y



def run_preprocessing(df_raw):
    lookback = CONFIG["model"]["lookback"]
    horizon  = CONFIG["model"]["horizon"]

    df = resample_3h(df_raw)        
    df = add_cyclic_features(df)   
    df = remove_missing(df)

    train, test = train_test_split_ts(df)

    scaled_train, scaled_test, scaler = scale_data(train, test)

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
        "scaled_data": np.vstack([scaled_train, scaled_test]), 
        "X_train": X_train,
        "y_train": y_train,
        "X_test":  X_test,
        "y_test":  y_test,
    }

def prepare_inference_input(df_raw, scaler, lookback):
    df     = resample_3h(df_raw)
    df     = add_cyclic_features(df)
    df     = remove_missing(df)
    scaled = scaler.transform(df)         
    last_sequence = scaled[-lookback:]
    return df, scaled, last_sequence