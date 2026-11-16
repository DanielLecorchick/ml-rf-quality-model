import csv
import numpy as np
import pytest

from rf_quality.dataset import WindowDataset
from rf_quality.evaluation import station_disjoint_split
from rf_quality.features import FEATURE_NAMES, extract_features
from rf_quality.labels import create_label_template, apply_labels
from rf_quality.synthetic import generate_synthetic


def test_feature_values_and_shapes():
    n = 600
    x = np.stack([np.cos(np.arange(n) / 8), np.sin(np.arange(n) / 8)])
    features = extract_features(x)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()
    assert 0 <= features[FEATURE_NAMES.index("spectral_entropy")] <= 1.00001


def test_synthetic_roundtrip_and_station_split(tmp_path):
    ds = generate_synthetic(stations=7, per_station=8, length=100)
    path = ds.save(tmp_path / "test.npz")
    loaded = WindowDataset.load(path)
    assert loaded.origin == "synthetic"
    assert loaded.signals.shape == (56, 2, 100)
    np.testing.assert_array_equal(loaded.labels, ds.labels)
    tr, te = station_disjoint_split(loaded)
    assert set(loaded.station_ids[tr]).isdisjoint(set(loaded.station_ids[te]))
    assert set(loaded.labels[tr]) == {0, 1}
    assert set(loaded.labels[te]) == {0, 1}


def test_label_csv(tmp_path):
    ds = generate_synthetic(stations=3, per_station=5, length=64)
    ds.labels[:] = -1
    filename = tmp_path / "label.csv"
    create_label_template(ds, filename)
    with filename.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert all(x["label"] == "" for x in rows)
    rows[0]["label"] = "1"
    rows[1]["label"] = "0"
    with filename.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    updated = apply_labels(ds, filename)
    assert updated.labels[:3].tolist() == [1, 0, -1]
    assert np.all(ds.labels == -1)
    rows[2]["label"] = "uncertain"
    with filename.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        apply_labels(ds, filename)


def test_unlabeled_data_rejected_by_evaluator():
    ds = generate_synthetic(stations=3, per_station=5)
    ds.labels[:] = -1
    with pytest.raises(ValueError, match="Need >=3 stations"):
        station_disjoint_split(ds)
