import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from src.data.database import load_weather_data, load_predictions

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "monitoring" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_monitoring_report(model_name: str = None, date: str = None) -> Path:

    obs   = load_weather_data()                      
    preds = load_predictions(model_name=model_name)    

    preds["timestamp"] = pd.to_datetime(preds["timestamp"])

    if date:
        preds["created_at"] = pd.to_datetime(preds["created_at"])
        preds = preds[preds["created_at"].dt.date == pd.to_datetime(date).date()]

    if preds.empty:
        raise ValueError(f"Aucune prédiction trouvée (model_name={model_name}, date={date})")

    merged = preds.merge(
        obs.rename(columns={"temperature": "observed"}),
        left_on="timestamp",
        right_index=True,
        how="inner"
    )

    if merged.empty:
        raise ValueError("Aucune observation réelle disponible pour la période des prédictions.")

    errors = merged["predicted_value"] - merged["observed"]
    mae    = errors.abs().mean()
    rmse   = (errors**2).mean()**0.5

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(merged["timestamp"], merged["observed"],
             label="Observations réelles", color="#3ecf8e", linewidth=2)
    ax1.plot(merged["timestamp"], merged["predicted_value"],
             label="Prédictions", color="#7c6fff", linewidth=2, linestyle="--")
    ax1.fill_between(merged["timestamp"],
                     merged["predicted_value"], merged["observed"],
                     alpha=0.12, color="#f0a04b", label="Écart")

    title = f"Monitoring — Prédictions vs Observations"
    if model_name:
        title += f"  |  modèle : {model_name}"
    ax1.set_title(f"{title}\nMAE = {mae:.2f}°C   RMSE = {rmse:.2f}°C", fontsize=11)
    ax1.set_ylabel("Température (°C)")
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(axis="y", alpha=0.2)

    ax2.bar(merged["timestamp"], errors,
            color=errors.apply(lambda e: "#f06070" if e > 0 else "#3ecf8e"),
            width=0.08, alpha=0.8)
    ax2.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Erreur (°C)")
    ax2.set_xlabel("Datetime")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.2)

    plt.tight_layout()

    suffix = f"{model_name or 'all'}_{date or 'all'}"
    output_path = OUTPUT_DIR / f"monitoring_{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[monitoring] MAE={mae:.2f}°C | RMSE={rmse:.2f}°C | graphique → {output_path}")
    return output_path


def generate_monitoring_data(model_name=None, date=None) -> dict:
    obs   = load_weather_data()
    preds = load_predictions(model_name=model_name)

    preds["timestamp"] = pd.to_datetime(preds["timestamp"])
    if date:
        preds["created_at"] = pd.to_datetime(preds["created_at"])
        preds = preds[preds["created_at"].dt.date == pd.to_datetime(date).date()]

    if preds.empty:
        raise ValueError("Aucune prédiction trouvée.")

    merged = preds.merge(
        obs.rename(columns={"temperature": "observed"}),
        left_on="timestamp", right_index=True, how="inner"
    )

    if merged.empty:
        raise ValueError("Aucune observation réelle disponible pour la période des prédictions.")

    errors = merged["predicted_value"] - merged["observed"]

    return {
        "mae":        round(errors.abs().mean(), 2),
        "rmse":       round((errors**2).mean()**0.5, 2),
        "model_name": model_name,
        "date":       date,
        "data":       merged[["timestamp", "predicted_value", "observed"]]
                      .assign(timestamp=merged["timestamp"].astype(str))
                      .to_dict(orient="records")
    }


if __name__ == "__main__":
    generate_monitoring_report()
