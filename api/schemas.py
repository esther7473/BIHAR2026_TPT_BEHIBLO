from pydantic import BaseModel
from datetime import datetime

class PredictionOut(BaseModel):
    id:              int
    model_name:      str
    timestamp:       datetime
    horizon:         int
    predicted_value: float
    created_at:      datetime

class CombinedOut(BaseModel):
    timestamp:       datetime
    horizon:         int
    predicted_value: float
    model_name:      str
    created_at:      datetime
    observed_temp:   float | None

class VersionOut(BaseModel):
    software_version: str = "0.0.0"
    model_name:       str
    model_version:    str | int

