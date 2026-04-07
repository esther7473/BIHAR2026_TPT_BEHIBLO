from src.data.download_data import fetch_recent_data
from src.training.train import train
from src.inference.inference import run_inference
from src.common.common import CONFIG

def main(end_date=None):
    end_date = end_date or CONFIG["api"]["end_date"]

    print("\n========== 1. FETCH ==========")
    fetch_recent_data(end_date=end_date)

    print("\n========== 2. TRAIN ==========")
    train()

    print("\n========== 3. INFERENCE ==========")
    timestamps, predictions = run_inference()
    print(f" {len(predictions)} prédictions générées : {timestamps[0]} --- {timestamps[-1]}")

    print("\n========== 1. FETCH PREDICTED ==========")
    fetch_recent_data(end_date=timestamps[-1])

if __name__ == "__main__":
    main(end_date="2026-03-03")