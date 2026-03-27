import sqlite3
import pandas as pd
from pathlib import Path
from src.common.common import CONFIG

# Chemin absolu basé sur l'emplacement de ce fichier (src/data/database.py)
_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = (_ROOT / CONFIG["paths"]["db_path"]).resolve()

# S'assurer que le dossier existe
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)  # garantit que les tables existent à chaque connexion
    return conn


def init_db(conn=None):
    """Crée les tables si elles n'existent pas. Accepte une connexion existante ou en crée une."""
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