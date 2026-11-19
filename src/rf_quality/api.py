"""Loopback-only FastAPI scoring service backed by a trusted model artifact.

No external Vector API requests are issued; alerts are stored in local SQLite.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from .alerts import AlertPolicy, LocalStationMonitor
from .artifacts import load_model_bundle, predict_bundle
from .features import FEATURE_NAMES, extract_features


class IQObservation(BaseModel):
    station_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_.:-]+$")
    window_id: str | None = Field(default=None, max_length=256)
    i: list[float] = Field(min_length=8)
    q: list[float] = Field(min_length=8)

    @field_validator("i", "q")
    @classmethod
    def finite(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(v) for v in values):
            raise ValueError("I/Q samples must be finite")
        return values


class IQBatch(BaseModel):
    observations: list[IQObservation] = Field(min_length=1, max_length=64)


def create_app(model_directory: Path, *, alert_db: Path = Path("results/local_alerts.sqlite"),
               policy: AlertPolicy | None = None) -> FastAPI:
    """Load once at startup; validate exact feature and signal schemas per request."""
    model, manifest = load_model_bundle(Path(model_directory))
    if tuple(manifest["feature_names"]) != FEATURE_NAMES:
        raise ValueError("The trained feature schema does not match this application version")
    monitor = LocalStationMonitor(alert_db, policy)
    app = FastAPI(title="PSWS RF Quality (local prototype)", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": manifest["name"], "alerts": "local-only"}

    @app.get("/model")
    def model_info():
        return {"name": manifest["name"], "project_version": manifest["project_version"],
                "schema_version": manifest["schema_version"],
                "signal_length": manifest["signal_length"],
                "feature_names": manifest["feature_names"],
                "decision_threshold": manifest["decision_threshold"],
                "score_is_probability": manifest["score_is_probability"]}

    def score_observations(observations: list[IQObservation]) -> list[dict]:
        expected = int(manifest["signal_length"])
        if any(len(obs.i) != expected or len(obs.q) != expected for obs in observations):
            raise HTTPException(status_code=422,
                                detail=f"Expected exactly {expected} I and Q samples per observation")
        signals = np.asarray([[obs.i, obs.q] for obs in observations], dtype=np.float32)
        if not np.isfinite(signals).all():
            raise HTTPException(status_code=422, detail="I/Q must remain finite as float32")
        features = np.stack([extract_features(x) for x in signals])
        result = predict_bundle(model, manifest, signals, features, FEATURE_NAMES)
        output = []
        for obs, score, pred in zip(observations, result.score, result.predicted):
            station = monitor.record(obs.station_id, float(score), bool(pred), obs.window_id)
            output.append({"station_id": obs.station_id, "window_id": obs.window_id,
                           "quality": "poor" if pred else "acceptable", "poor_quality": bool(pred),
                           "score": float(score), "score_is_probability": result.is_probability,
                           "threshold": float(model.threshold),
                           "features": {key: float(value) for key, value in zip(FEATURE_NAMES, features[len(output)])},
                           "confidence_score": (float(max(score, 1 - score))
                                                if result.is_probability else None),
                           "confidence_calibrated": False,
                           "station": station})
        return output

    @app.post("/predict")
    def predict(observation: IQObservation):
        return score_observations([observation])[0]

    @app.post("/predict/batch")
    def predict_batch(batch: IQBatch):
        return {"predictions": score_observations(batch.observations)}

    @app.get("/stations/{station_id}")
    def station(station_id: str):
        return monitor.station(station_id)

    @app.get("/alerts")
    def alerts(limit: Annotated[int, Query(ge=1, le=500)] = 100):
        return {"delivery": "local-sqlite-only", "alerts": monitor.alerts(limit)}

    return app
