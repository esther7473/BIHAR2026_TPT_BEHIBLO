import sqlite3
import pandas as pd
from pathlib import Path
from src.common.common import CONFIG

_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = (_ROOT / CONFIG["paths"]["db_path"]).resolve()


DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn) 
    return conn


def init_db(conn=None):
    close_after = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        close_after = True

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            timestamp   TEXT PRIMARY KEY,
            temperature REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name      TEXT,
            timestamp       TEXT,
            horizon         INTEGER,
            predicted_value REAL,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    # cursor.execute("""
    #     CREATE TABLE IF NOT EXISTS model_registry (
    #         id           INTEGER PRIMARY KEY AUTOINCREMENT,
    #         model_name   TEXT,
    #         version      INTEGER,
    #         run_id       TEXT,
    #         rmse         REAL,
    #         promoted_at  TEXT DEFAULT (datetime('now')),
    #         is_champion  INTEGER DEFAULT 1
    #     )
    # """)

    conn.commit()
    if close_after:
        conn.close()


def save_weather_data(df):
    conn = get_connection()
    df.to_sql("weather_data", conn, if_exists="append", index=True)
    conn.close()


def save_predictions(model_name, timestamps, predictions, horizon):
    conn = get_connection()
    df = pd.DataFrame({
        "model_name":      model_name,
        "timestamp":       timestamps,
        "horizon":         horizon,
        "predicted_value": predictions
    })
    df.to_sql("predictions", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


def load_weather_data():
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM weather_data",
        conn,
        index_col="timestamp",
        parse_dates=["timestamp"]
    )
    conn.close()
    return df


def load_predictions(model_name=None):
    conn = get_connection()
    query = "SELECT * FROM predictions"
    if model_name:
        query += f" WHERE model_name = '{model_name}'"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def get_last_timestamp():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(timestamp) FROM weather_data")
    result = cursor.fetchone()[0]
    conn.close()
    return pd.to_datetime(result) if result else None


def get_context(n_points):
    """
    Récupère les n_points dernières observations depuis la BDD
    pour construire le contexte d'entrée des modèles
    """
    conn  = get_connection()
    query = f"""
        SELECT timestamp, temperature 
        FROM weather_data 
        ORDER BY timestamp DESC 
        LIMIT {n_points}
    """
    df = pd.read_sql(query, conn, index_col="timestamp", parse_dates=["timestamp"])
    conn.close()
    return df.sort_index()



def save_model_version(model_name: str, version: int, run_id: str, rmse: float):
    conn = get_connection()
    conn.execute("UPDATE model_registry SET is_champion = 0 WHERE model_name = ?", [model_name])
    conn.execute("""
        INSERT INTO model_registry (model_name, version, run_id, rmse, is_champion)
        VALUES (?, ?, ?, ?, 1)
    """, [model_name, version, run_id, rmse])
    conn.commit()
    conn.close()


# def get_champion_version(model_name: str):
#     conn = get_connection()
#     conn.row_factory = sqlite3.Row
#     row = conn.execute("""
#         SELECT * FROM model_registry
#         WHERE model_name = ? AND is_champion = 1
#     """, [model_name]).fetchone()
#     conn.close()
#     return dict(row) if row else None