"""Station-disjoint evaluation of the same observations across all five models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from .dataset import WindowDataset
from .models.adapters import AVAILABLE_MODELS, make_model


def station_disjoint_split(data: WindowDataset, *, seed: int = 42, test_size: float = 0.25):
    """Return labeled train/test indices with NO shared stations; require both classes in both."""
    idx = np.flatnonzero(data.labels >= 0)
    labels = data.labels[idx]
    groups = data.station_ids[idx]
    if len(np.unique(groups)) < 3:
        raise ValueError("Need >=3 stations with labels for station-disjoint testing")
    for offset in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + offset)
        train_local, test_local = next(splitter.split(idx, labels, groups))
        train_idx, test_idx = idx[train_local], idx[test_local]
        if (len(np.unique(data.labels[train_idx])) == 2
                and len(np.unique(data.labels[test_idx])) == 2):
            assert not (set(data.station_ids[train_idx]) & set(data.station_ids[test_idx]))
            return train_idx, test_idx
    raise ValueError("Could not find station-disjoint train/test splits containing both labels. "
                     "Collect labels from more stations or adjust your sampling strategy")


def benchmark(
    data: WindowDataset, models: list[str], output_dir: Path, *, seed: int = 42,
    epochs: int = 8, rocket_kernels: int = 2000, continue_on_missing: bool = True,
) -> list[dict]:
    if any(m not in AVAILABLE_MODELS for m in models):
        raise ValueError(f"Model names must be in {AVAILABLE_MODELS}")
    train_idx, test_idx = station_disjoint_split(data, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "origin": data.origin,
        "seed": seed,
        "train_count": int(len(train_idx)), "test_count": int(len(test_idx)),
        "train_stations": sorted(set(map(str, data.station_ids[train_idx]))),
        "test_stations": sorted(set(map(str, data.station_ids[test_idx]))),
        "positive_class": "poor-quality observation",
        "labels": {"0": "acceptable", "1": "poor", "-1": "unknown"},
    }
    (output_dir / "split.json").write_text(json.dumps(summary, indent=2) + "\n")
    reports: list[dict] = []
    for name in models:
        try:
            model = make_model(name, seed=seed, epochs=epochs, rocket_kernels=rocket_kernels)
        except (RuntimeError, ImportError) as exc:
            if not continue_on_missing:
                raise
            reports.append({"model": name, "status": "missing_dependency", "message": str(exc)})
            continue
        started = time.perf_counter()
        model.fit(data.signals[train_idx], data.features[train_idx], data.labels[train_idx])
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        result = model.predict(data.signals[test_idx], data.features[test_idx])
        predict_seconds = time.perf_counter() - started
        y = data.labels[test_idx]
        if len(result.score) != len(y) or not np.isfinite(result.score).all():
            raise ValueError(f"{name}: invalid inference output")
        tn, fp, fn, tp = confusion_matrix(y, result.predicted, labels=[0, 1]).ravel()
        report = {
            "model": name, "status": "ok", "test_count": int(len(y)),
            "accuracy": float(accuracy_score(y, result.predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, result.predicted)),
            "precision_poor": float(precision_score(y, result.predicted, zero_division=0)),
            "recall_poor": float(recall_score(y, result.predicted, zero_division=0)),
            "f1_poor": float(f1_score(y, result.predicted, zero_division=0)),
            "average_precision_poor": float(average_precision_score(y, result.score)),
            "roc_auc_poor": float(roc_auc_score(y, result.score)),
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
            "fit_seconds": round(fit_seconds, 5),
            "predict_ms_per_window": round(predict_seconds * 1000 / len(y), 6),
            "score_is_probability": result.is_probability,
            "decision_threshold": float(0.5 if result.is_probability else model.threshold),
        }
        reports.append(report)
        with (output_dir / f"{name}_predictions.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["window_id", "station_id", "true_label", "predicted_label", "score"])
            for idx, pred, score in zip(test_idx, result.predicted, result.score):
                writer.writerow([data.window_ids[idx], data.station_ids[idx], data.labels[idx],
                                 int(pred), float(score)])
        (output_dir / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    fields = ["model", "status", "accuracy", "f1_poor", "recall_poor", "average_precision_poor",
              "fit_seconds", "predict_ms_per_window", "message"]
    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)
    return reports
