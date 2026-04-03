import os
import sqlite3
import logging
from fastapi import FastAPI, Depends, Query, HTTPException
from src.data.database import get_connection
from src.common.common import CONFIG, ROOT_DIR
from fastapi.responses import RedirectResponse
from api.schemas import PredictionOut, CombinedOut, VersionOut
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
from src.monitoring.monitoring import  generate_monitoring_data
from src.inference.get_run import get_latest_run_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


app = FastAPI(title="Weather Forecast API", version="0.0.0")
Instrumentator().instrument(app).expose(app)

mae_gauge  = Gauge("model_mae",  "MAE entre prédictions et observations", ["model_name"])
rmse_gauge = Gauge("model_rmse", "RMSE entre prédictions et observations", ["model_name"])


@app.get("/")
def root():
    return RedirectResponse(url="/docs")

def get_db():
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


predictions_counter = Counter(
    "predictions_total",
    "Nombre de prédictions générées",
    ["model_name"]
)

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
    
    predictions_counter.labels(model_name=model_name or "all").inc()

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
        model_metadata = get_latest_run_metadata()
        print(model_metadata)
        if not model_metadata:
            raise HTTPException(status_code=404, detail="Aucune version trouvé.")

        return {
            "model_name":    model_metadata["name"],
            "model_version": model_metadata["version"],
            "software_version": "0.0.0"

        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du modèle : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/monitoring")
def monitoring(model_name: str = None, date: str = None):
    try:
        mae_gauge.labels(model_name=label).set(result["mae"])
        rmse_gauge.labels(model_name=label).set(result["rmse"])

        for row in result["data"]:
            error = row["predicted_value"] - row["observed"]
            prediction_error.labels(model_name=label).observe(error)

        return generate_monitoring_data(model_name=model_name, date=date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))