from datetime import datetime, timedelta, timezone

import pytest

from rf_quality.alerts import AlertPolicy, LocalStationMonitor
from rf_quality.artifacts import save_model_bundle
from rf_quality.models.adapters import XGBoostRF
from rf_quality.synthetic import generate_synthetic


def test_local_alerts_dedupe_and_cooldown(tmp_path):
    monitor = LocalStationMonitor(tmp_path / "alerts.sqlite", AlertPolicy(
        min_windows=3, poor_fraction=2 / 3, history_windows=5, cooldown_seconds=60))
    t = datetime(2026, 11, 16, tzinfo=timezone.utc)
    assert monitor.record("A", 0.8, True, "a", t)["new_alert"] is None
    assert monitor.record("A", 0.7, True, "b", t + timedelta(seconds=1))["new_alert"] is None
    third = monitor.record("A", 0.2, False, "c", t + timedelta(seconds=2))
    assert third["new_alert"] is not None
    assert third["flagged"] is True
    assert monitor.record("A", 0.2, False, "c", t + timedelta(seconds=2))["duplicate"]
    assert len(monitor.alerts()) == 1
    monitor.record("A", 0.9, True, "d", t + timedelta(seconds=3))
    assert len(monitor.alerts()) == 1  # cooldown, even if condition persists
    monitor.record("A", 0.8, True, "e", t + timedelta(seconds=75))
    assert len(monitor.alerts()) == 2
    assert monitor.station("B")["history_count"] == 0
    assert not monitor.record("B", 0.7, True, "a", t)["duplicate"]


def test_api_model_info_and_iq_predictions(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from rf_quality.api import create_app
    data = generate_synthetic(stations=5, per_station=8, length=64)
    model = XGBoostRF(seed=1, n_estimators=5)
    model.fit(data.signals, data.features, data.labels)
    artifact = save_model_bundle(model, "xgboost", data, tmp_path / "artifact",
                                 seed=1, parameters={"n_estimators": 5})
    api = TestClient(create_app(artifact, alert_db=tmp_path / "alerts.sqlite",
                                policy=AlertPolicy(min_windows=2, poor_fraction=0.5)))
    assert api.get("/health").json()["alerts"] == "local-only"
    assert api.get("/model").json()["signal_length"] == 64
    observation = {"station_id": "S000001", "window_id": "w1",
                   "i": data.signals[0, 0].tolist(), "q": data.signals[0, 1].tolist()}
    response = api.post("/predict", json=observation)
    assert response.status_code == 200, response.text
    assert response.json()["quality"] in {"acceptable", "poor"}
    assert api.post("/predict", json=observation).json()["station"]["duplicate"] is True
    assert api.get("/stations/S000001").json()["history_count"] == 1
    assert api.post("/predict", json={**observation, "q": [0.0]}).status_code == 422
    batch = api.post("/predict/batch", json={"observations": [{**observation, "window_id": "w2"}]})
    assert len(batch.json()["predictions"]) == 1
    assert api.get("/alerts").status_code == 200
    assert api.get("/alerts").json()["delivery"] == "local-sqlite-only"
