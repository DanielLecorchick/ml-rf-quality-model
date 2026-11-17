import json

import numpy as np
import pytest

from rf_quality.artifacts import load_model_bundle, predict_bundle
from rf_quality.dataset import WindowDataset
from rf_quality.experiments import choose_threshold, run_experiments, split_dataset
from rf_quality.synthetic import generate_synthetic


def test_station_three_way_no_leakage():
    data = generate_synthetic(stations=9, per_station=16, length=64)
    split = split_dataset(data, seed=42)
    stations = [set(data.station_ids[i]) for i in (split.train, split.validation, split.test)]
    assert not (stations[0] & stations[1] or stations[0] & stations[2] or stations[1] & stations[2])
    assert all(set(data.labels[i]) == {0, 1} for i in (split.train, split.validation, split.test))


def test_chronological_holdout_with_embargo():
    data = generate_synthetic(stations=4, per_station=24, length=64)
    split = split_dataset(data, mode="time", embargo_seconds=60)
    for station in np.unique(data.station_ids):
        tr = split.train[data.station_ids[split.train] == station]
        va = split.validation[data.station_ids[split.validation] == station]
        te = split.test[data.station_ids[split.test] == station]
        assert data.timestamps[tr].max() < data.timestamps[va].min()
        assert data.timestamps[va].max() < data.timestamps[te].min()
        assert data.timestamps[va].min() - data.timestamps[tr].max() > 60
        assert data.timestamps[te].min() - data.timestamps[va].max() > 60


def test_temporal_mode_requires_timestamps():
    data = generate_synthetic()
    data.timestamps = None
    with pytest.raises(ValueError, match="timestamps"):
        split_dataset(data, mode="time")


def test_threshold_search_does_not_need_test_labels():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.35, 0.7, 0.9])
    threshold, val = choose_threshold(y, scores)
    assert threshold == pytest.approx(0.7)
    assert val == 1.0
    high_precision_threshold, _ = choose_threshold(y, scores, metric="recall_at_precision")
    assert high_precision_threshold == pytest.approx(0.7)


def test_grid_artifacts_reload_and_figures(tmp_path):
    data = generate_synthetic(stations=7, per_station=12, length=64)
    results = run_experiments(
        data, ["xgboost", "isolation_forest"], tmp_path, seeds=[42, 43],
        grid={"xgboost": [{"max_depth": 2, "n_estimators": 10},
                          {"max_depth": 3, "n_estimators": 12}]},
    )
    assert len(results) == 4
    assert all(r["status"] == "ok" for r in results)
    assert (tmp_path / "model_comparison.png").is_file()
    assert (tmp_path / "comparison.md").is_file()
    assert (tmp_path / "seed-42" / "xgboost" / "feature_importance.png").is_file()
    assert len(json.loads((tmp_path / "seed-42" / "xgboost" / "trials.json").read_text())) == 2
    for name in ("xgboost", "isolation_forest"):
        artifact = tmp_path / "seed-42" / name / "artifact"
        model, manifest = load_model_bundle(artifact)
        assert manifest["feature_names"] == list(data.feature_names)
        assert manifest["signal_length"] == 64
        assert manifest["class_mapping"] == {"0": "acceptable", "1": "poor-quality"}
        result = predict_bundle(model, manifest, data.signals[:2], data.features[:2], data.feature_names)
        assert result.score.shape == (2,)
        with pytest.raises(ValueError, match="schema mismatch"):
            predict_bundle(model, manifest, data.signals[:2], data.features[:2], ["bad"])
    artifact = tmp_path / "seed-42" / "xgboost" / "artifact"
    artifact.joinpath("model.json").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_model_bundle(artifact)


def test_legacy_npz_without_timestamps_loads(tmp_path):
    data = generate_synthetic(stations=3, per_station=4, length=64)
    data.timestamps = None
    loaded = WindowDataset.load(data.save(tmp_path / "legacy.npz"))
    assert loaded.timestamps is None
