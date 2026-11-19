"""Focused tests for source reproducibility, visual review, nested ZIPs and alert outbox."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sqlite3
import zipfile

import numpy as np
import pytest

from rf_quality.alerts import AlertPolicy, LocalStationMonitor
from rf_quality.annotation import build_review_gallery, select_review_indices
from rf_quality.artifacts import load_model_bundle, predict_bundle, save_model_bundle
from rf_quality.delivery import dispatch_pending_alerts
from rf_quality.ingestion.archives import extract_observation_archives
from rf_quality.synthetic import generate_synthetic


def _zip(name: str, data: bytes) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(name, data)
    return output.getvalue()


def test_nested_zip_extracts_and_protects_paths(tmp_path):
    inner = _zip("S000028/OBS2026-01-01/ch0/raw.h5", b"test")
    outer = tmp_path / "outer.zip"
    outer.write_bytes(_zip("observation.zip", inner))
    dest = tmp_path / "extract"
    paths = extract_observation_archives(outer, dest)
    assert len(paths) == 2
    assert (dest / "observation" / "S000028" / "OBS2026-01-01" / "ch0" / "raw.h5").read_bytes() == b"test"
    with pytest.raises(ValueError, match="depth"):
        extract_observation_archives(outer, tmp_path / "depth-limit", max_depth=0)
    with pytest.raises(ValueError, match="size"):
        extract_observation_archives(outer, tmp_path / "size-limit", max_expanded_bytes=10)
    outer.write_bytes(_zip("observation.zip", _zip("../escape.txt", b"pwn")))
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        extract_observation_archives(outer, tmp_path / "malicious")
    assert not (tmp_path / "escape.txt").exists()


def test_review_gallery_spectrogram_and_csv_template(tmp_path):
    ds = generate_synthetic(stations=3, per_station=4, length=64)
    indices = select_review_indices(ds, 6)
    assert len(indices) == 6
    assert len(set(map(str, ds.station_ids[indices]))) == 3
    page = build_review_gallery(ds, tmp_path / "review", max_windows=3)
    assert page.is_file()
    assert "Export review.csv" in page.read_text()
    assert "PSWS RF quality review" in page.read_text()
    records = json.loads((page.parent / "review_manifest.json").read_text())
    assert len(records) == 3
    assert all((page.parent / row["plot"]).is_file() for row in records)
    assert (page.parent / "review_template_all.csv").is_file()


def test_inception_native_source_and_bundle_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from rf_quality.models.adapters import make_model
    ds = generate_synthetic(stations=3, per_station=4, length=64)
    model = make_model("inceptiontime", seed=11, params={"filters": 4, "epochs": 1})
    assert model.backend == "native"
    assert model.model(
        torch.zeros(2, 2, 64, device=model.device)
    ).shape == (2, 2)
    model._normalize_fit(ds.signals)
    model.threshold = .5
    artifact = save_model_bundle(model, "inceptiontime", ds, tmp_path / "native", seed=11,
                                 parameters={"filters": 4, "epochs": 1})
    recovered, manifest = load_model_bundle(artifact)
    assert manifest["parameters"]["backend"] == "native"
    before = model.score(ds.signals[:2], ds.features[:2])
    after = predict_bundle(recovered, manifest, ds.signals[:2], ds.features[:2], ds.feature_names).score
    np.testing.assert_allclose(before, after, rtol=1e-6)
    # Tamper model source fingerprint while keeping model weights intact.
    data = json.loads((artifact / "manifest.json").read_text())
    data["source_sha256"]["features.py"] = "f" * 64
    (artifact / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Project source changed"):
        load_model_bundle(artifact)


class FakeResponse:
    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self):
        self.events = []
    def post(self, url, json, headers, timeout):
        self.events.append((url, json, headers))
        return FakeResponse()


def test_alert_dispatch_dry_run_explicit_https_and_idempotency(tmp_path):
    file = tmp_path / "alerts.sqlite"
    monitor = LocalStationMonitor(file, AlertPolicy(min_windows=2, poor_fraction=.5,
                                                    history_windows=2, cooldown_seconds=1000))
    monitor.record("S000028", .9, True, "window-1")
    monitor.record("S000028", .8, True, "window-2")
    fake = FakeSession()
    dry = dispatch_pending_alerts(file, endpoint="https://example.org/alerts", session=fake)
    assert dry.pending == 1 and dry.dry_run
    assert not fake.events
    with pytest.raises(ValueError, match="HTTPS"):
        dispatch_pending_alerts(file, endpoint="http://example.org/alerts", token="secret",
                                confirm_send=True, session=fake)
    with pytest.raises(ValueError, match="token"):
        dispatch_pending_alerts(file, endpoint="https://example.org/alerts",
                                confirm_send=True, session=fake)
    sent = dispatch_pending_alerts(file, endpoint="https://example.org/alerts", token="test-secret",
                                   confirm_send=True, session=fake)
    assert (sent.pending, sent.delivered, sent.failed) == (1, 1, 0)
    assert fake.events[0][1]["station_id"] == "S000028"
    assert fake.events[0][2]["Idempotency-Key"] == "local-alert-1"
    again = dispatch_pending_alerts(file, endpoint="https://example.org/alerts", token="test-secret",
                                    confirm_send=True, session=fake)
    assert again.pending == 0 and len(fake.events) == 1
    with sqlite3.connect(file) as con:
        assert con.execute("SELECT state, attempts FROM alert_deliveries").fetchone() == ("delivered", 1)
