import pytest
import sqlite3
from unittest.mock import patch, MagicMock
import sys


prometheus_mock = MagicMock()
prometheus_mock.Instrumentator.return_value.instrument.return_value.expose.return_value = None

sys.modules.setdefault("prometheus_fastapi_instrumentator", prometheus_mock)
sys.modules.setdefault("prometheus_client", MagicMock())

from fastapi.testclient import TestClient
from api.main import app



@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE weather_data (
            timestamp   TEXT PRIMARY KEY,
            temperature REAL
        );
        CREATE TABLE predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name      TEXT,
            timestamp       TEXT,
            horizon         INTEGER,
            predicted_value REAL,
            created_at      TEXT DEFAULT (datetime('now'))
        );
        INSERT INTO weather_data VALUES ('2026-03-01 00:00:00', 8.5);
        INSERT INTO weather_data VALUES ('2026-03-01 03:00:00', 9.0);
        INSERT INTO predictions (model_name, timestamp, horizon, predicted_value)
            VALUES ('lstm_test', '2026-03-01 00:00:00', 24, 10.5);
        INSERT INTO predictions (model_name, timestamp, horizon, predicted_value)
            VALUES ('lstm_test', '2026-03-01 03:00:00', 24, 11.2);
    """)
    return conn


@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass 
    app.dependency_overrides[__import__("api.main", fromlist=["get_db"]).get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()



def test_get_predictions(client):
    response = client.get("/predictions")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["model_name"] == "lstm_test"


def test_get_predictions_filter_date(client):
    response = client.get("/predictions?date=2026-03-01 03:00:00")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_predictions_filter_model(client):
    response = client.get("/predictions?model_name=lstm_test")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_predictions_not_found(client):
    response = client.get("/predictions?model_name=inexistant")
    assert response.status_code == 404



def test_get_combined(client):
    response = client.get(
        "/predictions/combined",
        params={"start_date": "2026-03-01 00:00:00", "end_date": "2026-03-01 06:00:00"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["predicted_value"] == 10.5
    assert data[0]["observed_temp"]   == 8.5


def test_get_combined_not_found(client):
    response = client.get(
        "/predictions/combined",
        params={"start_date": "2020-01-01 00:00:00", "end_date": "2020-01-02 00:00:00"}
    )
    assert response.status_code == 404




def test_root_redirect(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in [302, 307]
    assert response.headers["location"] == "/docs"