import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.download_data import fetch_temperature, fetch_recent_data


def mock_api_response():
    """Simule une réponse de l'API open-meteo."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "hourly": {
            "time":          ["2025-01-01T00:00", "2025-01-01T01:00", "2025-01-01T02:00"],
            "temperature_2m": [5.0, 4.5, 4.0]
        }
    }
    return mock_resp



def test_fetch_temperature():
    with patch("src.data.download_data.requests.get", return_value=mock_api_response()), \
         patch("src.data.download_data.save_weather_data") as mock_save:

        df = fetch_temperature(start_date="2025-01-01", end_date="2025-01-02")

        assert len(df) == 3
        assert "temperature" in df.columns
        assert df.index.name == "timestamp"

        mock_save.assert_called_once()


def test_fetch_recent_data_bd_vide():
    with patch("src.data.download_data.requests.get", return_value=mock_api_response()), \
         patch("src.data.download_data.save_weather_data"), \
         patch("src.data.download_data.get_last_timestamp", return_value=None):

        df = fetch_recent_data(end_date="2025-01-02")
        assert df is not None
        assert len(df) == 3


def test_fetch_recent_data_up_to_date():
    with patch("src.data.download_data.get_last_timestamp",
               return_value=pd.Timestamp("2025-12-31")):

        result = fetch_recent_data(end_date="2025-01-01")
        assert result is None


def test_fetch_recent_data_upgrade():
    with patch("src.data.download_data.requests.get", return_value=mock_api_response()), \
         patch("src.data.download_data.save_weather_data"), \
         patch("src.data.download_data.get_last_timestamp",
               return_value=pd.Timestamp("2024-12-31")):

        df = fetch_recent_data(end_date="2025-01-02")
        assert df is not None