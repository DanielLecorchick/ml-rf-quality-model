import csv
import json

from rf_quality.evaluation import benchmark
from rf_quality.synthetic import generate_synthetic


def test_xgboost_and_isolation_forest_smoke(tmp_path):
    ds = generate_synthetic(stations=6, per_station=7, length=64, seed=41)
    results = benchmark(ds, ["xgboost", "isolation_forest"], tmp_path, seed=41)
    assert len(results) == 2
    assert all(result["status"] == "ok" for result in results)
    assert (tmp_path / "summary.csv").exists()
    split = json.loads((tmp_path / "split.json").read_text())
    assert not (set(split["train_stations"]) & set(split["test_stations"]))
    with (tmp_path / "xgboost_predictions.csv").open(newline="") as file:
        assert len(list(csv.DictReader(file))) == split["test_count"]
