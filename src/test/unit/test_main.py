import pytest
import sqlite3
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
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
    """Client de test avec BD injectée."""
    def override_get_db():
        try:
            yield db
        finally:
            pass  # ne pas fermer la BD de test

    app.dependency_overrides[__import__("api.main", fromlist=["get_db"]).get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


# ─────────────────────────────────────────
# Tests /predictions
# ─────────────────────────────────────────

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


# ─────────────────────────────────────────
# Tests /predictions/combined
# ─────────────────────────────────────────

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


# ─────────────────────────────────────────
# Tests /version
# ─────────────────────────────────────────

def test_get_version_champion(client):
    mock_mv = MagicMock()
    mock_mv.name    = "lstm_multioutput_v2"
    mock_mv.version = "3"
    mock_mv.run_id  = "abc123"

    with patch("api.main.MlflowClient") as mock_client:
        mock_client.return_value.get_model_version_by_alias.return_value = mock_mv

        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"]    == "lstm_multioutput_v2"
        assert data["model_version"] == "3"
        assert data["stage"]         == "champion"
        assert "software_version"    in data  

def test_get_version_no_champion(client):
    with patch("api.main.MlflowClient") as mock_client:
        mock_client.return_value.get_model_version_by_alias.side_effect = Exception("not found")

        response = client.get("/version")
        assert response.status_code == 404


# ─────────────────────────────────────────
# Test /
# ─────────────────────────────────────────

def test_root_redirect(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in [302, 307]
    assert response.headers["location"] == "/docs"