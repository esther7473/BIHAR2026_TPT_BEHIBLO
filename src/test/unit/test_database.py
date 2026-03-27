import pytest
import sqlite3
import pandas as pd
from unittest.mock import patch
from src.data.database import (
    init_db, save_weather_data, save_predictions,
    load_weather_data, load_predictions,
    get_last_timestamp, get_context
)

class NoCloseConnection:
    """Wrapper qui empêche close() de fermer la connexion."""
    def __init__(self, conn):
        self._conn = conn

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._conn, name)


@pytest.fixture
def db():
    conn    = sqlite3.connect(":memory:", check_same_thread=False)
    wrapped = NoCloseConnection(conn)

    # get_connection retourne toujours le même wrapper (avec init_db intégré)
    with patch("src.data.database.get_connection", return_value=wrapped):
        init_db(conn)  # 👈 on passe la vraie connexion pour initialiser les tables
        yield wrapped

    conn.close()


def test_init_db(db):
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    assert "weather_data" in tables
    assert "predictions"  in tables


def test_save_weather_data(db):
    with patch("src.data.database.get_connection", return_value=db):
        df = pd.DataFrame({
            "timestamp":   ["2025-01-02", "2025-01-04"],
            "temperature": [50.0, 2.0]
        }).set_index("timestamp")
        save_weather_data(df)

        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM weather_data")
        assert cursor.fetchone()[0] == 2


def test_save_predictions(db):
    with patch("src.data.database.get_connection", return_value=db):
        save_predictions(
            model_name  = "lstm_test",
            timestamps  = ["2026-03-01 00:00:00", "2026-03-01 03:00:00"],
            predictions = [10.5, 11.2],
            horizon     = 24
        )
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        assert cursor.fetchone()[0] == 2


def test_load_weather_data(db):
    with patch("src.data.database.get_connection", return_value=db):
        save_weather_data(pd.DataFrame({
            "timestamp":   ["2025-01-01", "2025-01-02"],
            "temperature": [10.0, 12.0]
        }).set_index("timestamp"))

        df = load_weather_data()
        assert len(df) == 2
        assert "temperature" in df.columns
        assert df.index.name == "timestamp"


def test_load_predictions(db):
    with patch("src.data.database.get_connection", return_value=db):
        save_predictions(
            model_name  = "lstm_test",
            timestamps  = ["2026-03-01 00:00:00", "2026-03-01 03:00:00"],
            predictions = [10.5, 11.2],
            horizon     = 24
        )

        df = load_predictions()
        assert len(df) == 2

        df_filtered = load_predictions(model_name="lstm_test")
        assert len(df_filtered) == 2
        assert all(df_filtered["model_name"] == "lstm_test")

        df_empty = load_predictions(model_name="inexistant")
        assert len(df_empty) == 0


def test_get_last_timestamp(db):
    with patch("src.data.database.get_connection", return_value=db):
        assert get_last_timestamp() is None

        save_weather_data(pd.DataFrame({
            "timestamp":   ["2025-01-01", "2025-01-02", "2025-01-03"],
            "temperature": [10.0, 12.0, 14.0]
        }).set_index("timestamp"))

        last = get_last_timestamp()
        assert last == pd.to_datetime("2025-01-03")


def test_get_context(db):
    with patch("src.data.database.get_connection", return_value=db):
        save_weather_data(pd.DataFrame({
            "timestamp":   ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "temperature": [10.0, 12.0, 14.0, 16.0]
        }).set_index("timestamp"))

        df = get_context(n_points=2)
        assert len(df) == 2
        assert list(df["temperature"]) == [14.0, 16.0]