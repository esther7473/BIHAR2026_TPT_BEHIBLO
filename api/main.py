# api/main.py
import os
import sqlite3
import logging
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, Depends, Query, HTTPException
from src.data.database import get_connection
from src.common.common import CONFIG, ROOT_DIR
from fastapi.responses import RedirectResponse
from api.schemas import PredictionOut, CombinedOut, VersionOut



# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Weather Forecast API", version="0.0.0")


@app.get("/")
def root():
    return RedirectResponse(url="/docs")

# ── Dependency ────────────────────────────────────────────────────────────────
def get_db():
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/predictions", response_model=list[PredictionOut])
def get_predictions(
    date:       str | None = Query(None, description="Filtre >= date (YYYY-MM-DD HH:MM)"),
    model_name: str | None = Query(None, description="Filtre par nom de modèle"),
    db = Depends(get_db)
):
    query, params = "SELECT * FROM predictions WHERE 1=1", []
    if date:
        query += " AND timestamp >= ?"
        params.append(date)
    if model_name:
        query += " AND model_name = ?"
        params.append(model_name)
    rows = db.execute(query + " ORDER BY timestamp", params).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="Aucune prédiction trouvée.")
    return [dict(r) for r in rows]


@app.get("/predictions/combined", response_model=list[CombinedOut])
def get_combined(
    start_date: str = Query(..., description="YYYY-MM-DD HH:MM"),
    end_date:   str = Query(..., description="YYYY-MM-DD HH:MM"),
    db = Depends(get_db)
):
    rows = db.execute("""
        SELECT 
            p.timestamp,
            p.horizon,
            p.predicted_value,
            p.model_name,
            p.created_at,
            w.temperature AS observed_temp
        FROM predictions p
        LEFT JOIN weather_data w 
            ON w.timestamp = p.timestamp  
        WHERE p.timestamp BETWEEN ? AND ?
        ORDER BY p.timestamp
    """, [start_date, end_date]).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="Aucune donnée pour cette période.")
    return [dict(r) for r in rows]


@app.get("/version", response_model=VersionOut)
def get_version():
    mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
    client     = MlflowClient()
    model_name = CONFIG["model"]["name"]

    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
        return {
            "model_name":    mv.name,
            "model_version": mv.version,
            "run_id":        mv.run_id,
            "stage":         "champion",
        }
    except Exception:
        raise HTTPException(status_code=404, detail="Aucun modèle champion enregistré dans MLflow.")