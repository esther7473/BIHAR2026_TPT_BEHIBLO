import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from common import CONFIG
from database import save_weather_data


def resample_3h(df):
    return df["temperature"].resample(CONFIG["data"]["resample_freq"]).mean().to_frame()


def remove_missing(df):
    return df.dropna()


def train_test_split_ts(df):
    test_size = CONFIG["data"]["test_size"]
    split_idx = int(len(df) * (1 - test_size))
    train     = df.iloc[:split_idx]
    test      = df.iloc[split_idx:]
    return train, test


def get_tscv():
    return TimeSeriesSplit(n_splits=CONFIG["data"]["n_splits"])


def run_preprocessing(df_raw):
    df          = resample_3h(df_raw)
    df          = remove_missing(df)
    train, test = train_test_split_ts(df)

    save_weather_data(df)

    print(f"Total  : {len(df)} points")
    print(f"Train  : {len(train)} points")
    print(f"Test   : {len(test)} points")

    return df, train, test