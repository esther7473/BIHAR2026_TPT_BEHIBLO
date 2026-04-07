import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import kpss
from tcn import TCN
from sklearn.model_selection import TimeSeriesSplit
from pandas.plotting import autocorrelation_plot



def inverse_transform_multi(preds, scaler, n_features):
    result = []

    for i in range(preds.shape[1]):
        temp = np.zeros((len(preds), n_features))
        temp[:, 0] = preds[:, i]
        inv = scaler.inverse_transform(temp)[:, 0]
        result.append(inv)

    return np.array(result).T


def plot_model_comparison(results, title="Comparaison des modèles"):
    df_results = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    x     = np.arange(len(df_results))
    width = 0.5

    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        bars = ax.bar(x, df_results[metric], width, color="steelblue", edgecolor="white")

        for bar, val in zip(bars, df_results[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}°C", ha="center", va="bottom", fontsize=10)

        ax.set_title(metric)
        ax.set_ylabel("°C")
        ax.set_xticks(x)
        ax.set_xticklabels(df_results["Modèle"], rotation=15, ha="right")
        ax.set_ylim(0, df_results[metric].max() * 1.2)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
    
def create_sequences_multistep(data, lookback, horizon):
    X, y = [], []

    for i in range(len(data) - lookback - horizon):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon, 0])

    return np.array(X), np.array(y)

def inverse_transform_safe(preds, scaler, reference):
    temp = np.tile(reference, (len(preds), 1))
    temp[:, 0] = preds.flatten()
    return scaler.inverse_transform(temp)[:, 0]



def create_sequences(data, lookback=8):
    X, y = [], []

    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback, 0])

    return np.array(X), np.array(y)



def plot_series(df, title, xlabel="x", ylabel="y"):
    plt.figure(figsize=(10, 5))
    plt.plot(df)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_baselines(series, test, forecasts, months=3):

    end_date = series.index[-1]
    start_date = end_date - pd.DateOffset(months=months)
    series_3m = series.loc[start_date:end_date]

    plt.figure(figsize=(12, 5))

    plt.plot(series_3m.index, series_3m.values, label="Série originale")

    plt.plot(test.index, test.values, label="Vrai futur (test)", linewidth=3)

    for name, forecast in forecasts.items():
        forecast = pd.Series(forecast, index=test.index)
        plt.plot(
            forecast.index,
            forecast.values,
            label=name,
            linestyle="--"
        )

    plt.title("Baselines vs série originale")
    plt.xlabel("Date")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def baseline_forecast(series, train_len, horizon, window=1, ma_window=24):

    mean_preds = []
    naive_preds = []
    ma_preds = []

    for i in range(train_len, train_len + horizon, window):

        train = series[:i]

        mean_forecast = np.repeat(train.mean(), window)
        naive_forecast = np.repeat(train.iloc[-1], window)
        ma_forecast = np.repeat(train.iloc[-ma_window:].mean(), window)

        mean_preds.extend(mean_forecast)
        naive_preds.extend(naive_forecast)
        ma_preds.extend(ma_forecast)

    test = series[train_len:train_len + horizon]

    results = {
        "mean_pred": np.array(mean_preds[:horizon]),
        "naive_pred": np.array(naive_preds[:horizon]),
        "ma_pred": np.array(ma_preds[:horizon]),
        "mae_mean": mean_absolute_error(test, mean_preds[:horizon]),
        "mae_naive": mean_absolute_error(test, naive_preds[:horizon]),
        "mae_ma": mean_absolute_error(test, ma_preds[:horizon]),
        "rmse_mean":root_mean_squared_error(test, mean_preds[:horizon]),
        "rmse_naive":root_mean_squared_error(test, naive_preds[:horizon]),
        "rmse_ma":root_mean_squared_error(test, ma_preds[:horizon]),

    }

    return results