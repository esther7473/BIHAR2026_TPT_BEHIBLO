# api/main.py
import os
import sqlite3
import logging
from fastapi import FastAPI, Depends, Query, HTTPException
from src.data.database import get_connection
from src.common.common import CONFIG, ROOT_DIR
from fastapi.responses import RedirectResponse
from api.schemas import PredictionOut, CombinedOut, VersionOut
import pandas as pd



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
    try:
        model_path = CONFIG["paths"]["model_path"]

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Aucun modèle trouvé dans /models.")

        model_name    = CONFIG["model"]["name"]
        model_version = os.path.getmtime(model_path)
        model_date    = pd.Timestamp(model_version, unit="s").strftime("%Y-%m-%d %H:%M:%S")

        return {
            "model_name":    model_name,
            "model_version": model_date,
            # "run_id":        "local",
            # "stage":         "champion",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du modèle : {e}")
        raise HTTPException(status_code=500, detail=str(e))