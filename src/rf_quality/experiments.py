"""Train/validation/test experiments without tuning on the held-out test set."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from .artifacts import save_model_bundle
from .dataset import WindowDataset
from .models.adapters import AVAILABLE_MODELS, make_model


@dataclass(frozen=True)
class DataSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    mode: str


def _check_split(data: WindowDataset, split: DataSplit) -> DataSplit:
    if any(len(indices) == 0 for indices in (split.train, split.validation, split.test)):
        raise ValueError("A training, validation, or test partition is empty")
    for name in ("train", "validation", "test"):
        arr = getattr(split, name)
        if set(np.unique(data.labels[arr])) != {0, 1}:
            raise ValueError(f"{name} requires both classes; collect more diverse reviewed labels")
    if (set(split.train) & set(split.validation) or set(split.train) & set(split.test)
            or set(split.validation) & set(split.test)):
        raise ValueError("Overlapping split indices")
    if split.mode == "station":
        station_sets = [set(data.station_ids[x]) for x in (split.train, split.validation, split.test)]
        if any(station_sets[i] & station_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Station leakage between train, validation and test")
    return split


def split_dataset(
    data: WindowDataset, *, seed: int = 42, mode: str = "station",
    validation_fraction: float = 0.2, test_fraction: float = 0.2,
    embargo_seconds: float = 0.0,
) -> DataSplit:
    """Station-disjoint or per-station chronological three-way partition.

    Only manually labeled observations participate. Synthetic labels are useful
    for testing the harness but not for scientific evaluation.
    """
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 0.8):
        raise ValueError("Validation/test fractions must be positive and leave >=20% train")
    if embargo_seconds < 0:
        raise ValueError("embargo_seconds must be >=0")
    ids = np.flatnonzero(data.labels >= 0)
    if not len(ids):
        raise ValueError("There are no reviewed labels")
    if mode == "station":
        if len(np.unique(data.station_ids[ids])) < 3:
            raise ValueError("At least three labeled stations are required for station holdout")
        for attempt in range(300):
            first = GroupShuffleSplit(n_splits=1, test_size=test_fraction,
                                      random_state=seed + attempt)
            tv_local, te_local = next(first.split(ids, data.labels[ids], data.station_ids[ids]))
            tv = ids[tv_local]
            te = ids[te_local]
            second = GroupShuffleSplit(n_splits=1,
                                       test_size=validation_fraction / (1 - test_fraction),
                                       random_state=seed * 1009 + attempt)
            tr_local, va_local = next(second.split(tv, data.labels[tv], data.station_ids[tv]))
            try:
                return _check_split(data, DataSplit(tv[tr_local], tv[va_local], te, mode))
            except ValueError:
                continue
        raise ValueError("Cannot construct three disjoint station partitions with both labels. "
                         "Collect reviewed examples across additional stations")
    if mode == "time":
        if data.timestamps is None:
            raise ValueError("Temporal holdout requires window_start timestamps in the NPZ")
        train, val, test = [], [], []
        for station in np.unique(data.station_ids[ids]):
            station_ids = ids[data.station_ids[ids] == station]
            station_ids = station_ids[np.argsort(data.timestamps[station_ids], kind="stable")]
            n = len(station_ids)
            cut_train = int(np.floor(n * (1 - validation_fraction - test_fraction)))
            cut_val = int(np.floor(n * (1 - test_fraction)))
            if not (0 < cut_train < cut_val < n):
                continue
            tr, va, te = station_ids[:cut_train], station_ids[cut_train:cut_val], station_ids[cut_val:]
            if embargo_seconds:
                first_val_time = data.timestamps[va[0]]
                first_test_time = data.timestamps[te[0]]
                tr = tr[data.timestamps[tr] < first_val_time - embargo_seconds]
                va = va[(data.timestamps[va] >= first_val_time)
                        & (data.timestamps[va] < first_test_time - embargo_seconds)]
            train.extend(tr)
            val.extend(va)
            test.extend(te)
        split = DataSplit(np.array(train, dtype=int), np.array(val, dtype=int),
                          np.array(test, dtype=int), mode)
        return _check_split(data, split)
    raise ValueError("Split mode must be 'station' or 'time'")


def choose_threshold(y: np.ndarray, scores: np.ndarray, *, metric: str = "f1",
                     min_precision: float = 0.8) -> tuple[float, float]:
    """Tune a decision threshold on validation scores ONLY (higher means poorer)."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) != len(scores) or not np.isfinite(scores).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold optimization needs finite scores and both classes")
    if metric not in ("f1", "balanced_accuracy", "recall_at_precision"):
        raise ValueError("Unknown threshold metric")
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0,1]")
    thresholds = np.unique(scores)
    if len(thresholds) > 512:
        thresholds = np.unique(np.quantile(thresholds, np.linspace(0, 1, 512)))
    thresholds = np.r_[thresholds, np.nextafter(scores.max(), np.inf)]
    best: tuple[float, float, float] | None = None
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        if metric == "f1":
            value = f1_score(y, pred, zero_division=0)
        elif metric == "balanced_accuracy":
            value = balanced_accuracy_score(y, pred)
        else:
            value = recall if precision >= min_precision else -1.0
        candidate = (float(value), float(precision), float(threshold))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    if best[0] < 0:
        raise ValueError("No threshold reaches the requested minimum precision")
    return best[2], best[0]


def calculate_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_poor": float(precision_score(y, pred, zero_division=0)),
        "recall_poor": float(recall_score(y, pred, zero_division=0)),
        "f1_poor": float(f1_score(y, pred, zero_division=0)),
        "average_precision_poor": float(average_precision_score(y, scores)),
        "roc_auc_poor": float(roc_auc_score(y, scores)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _plot_results(model_name: str, y: np.ndarray, scores: np.ndarray, threshold: float,
                  output: Path, model: Any, feature_names: tuple[str, ...]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    pred = np.asarray(scores) >= threshold
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=[0, 1],
                                             display_labels=["Acceptable", "Poor"], ax=ax,
                                             colorbar=False)
    ax.set_title(f"{model_name} — held-out test")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y, scores)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(recall, precision, label=f"AP={average_precision_score(y, scores):.3f}")
    ax.set(xlabel="Recall (poor quality)", ylabel="Precision (poor quality)",
           title=f"{model_name} — test precision-recall", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "precision_recall.png", dpi=150)
    plt.close(fig)

    if model_name == "xgboost" and hasattr(model.model, "feature_importances_"):
        importance = np.asarray(model.model.feature_importances_)
        order = np.argsort(importance)[::-1][:18][::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(np.array(feature_names)[order], importance[order])
        ax.set(title="XGBoost feature importance (fitted model)", xlabel="Importance")
        fig.tight_layout()
        fig.savefig(output / "feature_importance.png", dpi=150)
        plt.close(fig)


def _csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(
    data: WindowDataset, models: list[str], output: Path, *,
    seeds: list[int] | tuple[int, ...] = (42,), split_mode: str = "station",
    validation_fraction: float = 0.2, test_fraction: float = 0.2,
    embargo_seconds: float = 0.0,
    grid: dict[str, list[dict[str, Any]]] | None = None,
    epochs: int = 8, rocket_kernels: int = 2000,
    threshold_metric: str = "f1", min_precision: float = 0.8,
    save_models: bool = True, strict: bool = False,
) -> list[dict[str, Any]]:
    """Select model hyperparams and threshold on validation; report test once per seed."""
    if any(x not in AVAILABLE_MODELS for x in models):
        raise ValueError("Unknown model name")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct random seeds")
    if grid is not None and (not isinstance(grid, dict) or set(grid) - set(AVAILABLE_MODELS)):
        raise ValueError("Grid keys must be known model names")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        split = split_dataset(data, seed=seed, mode=split_mode,
                              validation_fraction=validation_fraction,
                              test_fraction=test_fraction, embargo_seconds=embargo_seconds)
        root = output / f"seed-{seed}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "split.json").write_text(json.dumps({
            "seed": seed, "mode": split_mode, "embargo_seconds": embargo_seconds,
            "validation_fraction": validation_fraction, "test_fraction": test_fraction,
            "train_count": len(split.train), "validation_count": len(split.validation),
            "test_count": len(split.test),
            "train_class_counts": np.bincount(data.labels[split.train], minlength=2).tolist(),
            "validation_class_counts": np.bincount(data.labels[split.validation], minlength=2).tolist(),
            "test_class_counts": np.bincount(data.labels[split.test], minlength=2).tolist(),
            "train_stations": sorted(set(map(str, data.station_ids[split.train]))),
            "validation_stations": sorted(set(map(str, data.station_ids[split.validation]))),
            "test_stations": sorted(set(map(str, data.station_ids[split.test]))),
            "train_window_ids": list(map(str, data.window_ids[split.train])),
            "validation_window_ids": list(map(str, data.window_ids[split.validation])),
            "test_window_ids": list(map(str, data.window_ids[split.test])),
        }, indent=2) + "\n")
        for name in models:
            choices = (grid or {}).get(name, [{}])
            if not isinstance(choices, list) or not choices or not all(isinstance(c, dict) for c in choices):
                raise ValueError(f"Grid for {name} must contain a nonempty list of dicts")
            best: dict[str, Any] | None = None
            trials = []
            for trial_no, parameters in enumerate(choices):
                try:
                    model = make_model(name, seed=seed, epochs=epochs,
                                       rocket_kernels=rocket_kernels, params=parameters)
                except (ImportError, RuntimeError) as exc:
                    if strict:
                        raise
                    best = None
                    trials.append({"status": "missing_dependency", "message": str(exc)})
                    break
                start = time.perf_counter()
                model.fit(data.signals[split.train], data.features[split.train], data.labels[split.train])
                fit_seconds = time.perf_counter() - start
                validation_scores = np.asarray(model.score(data.signals[split.validation],
                                                           data.features[split.validation]), dtype=float)
                threshold, objective = choose_threshold(data.labels[split.validation], validation_scores,
                                                       metric=threshold_metric, min_precision=min_precision)
                vm = calculate_metrics(data.labels[split.validation], validation_scores, threshold)
                trials.append({"trial": trial_no, "parameters": parameters, "threshold": threshold,
                               "objective": objective, "fit_seconds": round(fit_seconds, 4),
                               "validation_metrics": vm, "status": "ok"})
                candidate = {"objective": objective, "validation_ap": vm["average_precision_poor"],
                             "model": model, "parameters": parameters, "threshold": threshold,
                             "trial": trial_no, "fit_seconds": fit_seconds, "validation_metrics": vm}
                if best is None or (candidate["objective"], candidate["validation_ap"]) > (
                        best["objective"], best["validation_ap"]):
                    best = candidate
            model_root = root / name
            model_root.mkdir(parents=True, exist_ok=True)
            (model_root / "trials.json").write_text(json.dumps(trials, indent=2) + "\n")
            if best is None:
                summaries.append({"seed": seed, "model": name, "status": "missing_dependency",
                                  "message": trials[-1]["message"]})
                continue
            fitted = best["model"]
            fitted.threshold = best["threshold"]
            start = time.perf_counter()
            test_scores = np.asarray(fitted.score(data.signals[split.test],
                                                  data.features[split.test]), dtype=float)
            predict_seconds = time.perf_counter() - start
            test_metrics = calculate_metrics(data.labels[split.test], test_scores, fitted.threshold)
            report = {
                "seed": seed, "model": name, "status": "ok", "split_mode": split_mode,
                "parameters": best["parameters"], "chosen_trial": best["trial"],
                "threshold": best["threshold"], "threshold_metric": threshold_metric,
                "validation_objective": best["objective"],
                "validation_metrics": best["validation_metrics"],
                **test_metrics,
                "fit_seconds": round(best["fit_seconds"], 4),
                "predict_ms_per_window": round(1000 * predict_seconds / len(split.test), 5),
                "score_is_probability": bool(fitted.probability),
            }
            if save_models:
                defaults = {"epochs": epochs} if name in ("inceptiontime", "autoencoder") else {}
                if name == "multirocket":
                    defaults["kernels"] = rocket_kernels
                defaults.update(best["parameters"])
                save_model_bundle(fitted, name, data, model_root / "artifact", seed=seed,
                                  parameters=defaults,
                                  metadata={"split_mode": split_mode,
                                            "training_windows": len(split.train),
                                            "validation_windows": len(split.validation),
                                            "validation_metrics": best["validation_metrics"]})
            _plot_results(name, data.labels[split.test], test_scores, fitted.threshold,
                          model_root, fitted, data.feature_names)
            _csv(model_root / "test_predictions.csv", [
                {"window_id": str(data.window_ids[i]), "station_id": str(data.station_ids[i]),
                 "true_label": int(data.labels[i]), "predicted_label": int(s >= fitted.threshold),
                 "score": float(s)} for i, s in zip(split.test, test_scores)
            ], ["window_id", "station_id", "true_label", "predicted_label", "score"])
            (model_root / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
            summaries.append(report)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    fields = ["seed", "model", "status", "split_mode", "f1_poor", "precision_poor",
              "recall_poor", "average_precision_poor", "roc_auc_poor", "threshold",
              "fit_seconds", "predict_ms_per_window", "message"]
    _csv(output / "summary.csv", summaries, fields)
    _write_comparison(summaries, output)
    return summaries


def _write_comparison(rows: list[dict[str, Any]], out: Path) -> None:
    successful = [r for r in rows if r["status"] == "ok"]
    lines = ["# RF quality model comparison", "",
             "Synthetic results, if used, are **not** evidence of PSWS performance.", "",
             "Parameters and thresholds were selected using validation sets. "
             "The test set was scored once for each selected model/seed.", "",
             "| Model | Runs | Test F1 (mean ± sd) | Test AP (mean ± sd) | Recall (mean) |",
             "|---|---:|---:|---:|---:|"]
    aggregates = []
    for name in sorted({r["model"] for r in successful}):
        group = [r for r in successful if r["model"] == name]
        f1 = np.array([r["f1_poor"] for r in group])
        ap = np.array([r["average_precision_poor"] for r in group])
        recall = np.array([r["recall_poor"] for r in group])
        aggregates.append({"model": name, "runs": len(group), "f1_mean": float(f1.mean()),
                           "f1_std": float(f1.std()), "ap_mean": float(ap.mean()),
                           "ap_std": float(ap.std()), "recall_mean": float(recall.mean())})
        lines.append(f"| {name} | {len(group)} | {f1.mean():.3f} ± {f1.std():.3f} | "
                     f"{ap.mean():.3f} ± {ap.std():.3f} | {recall.mean():.3f} |")
    missing = [r for r in rows if r["status"] != "ok"]
    if missing:
        lines.extend(["", "## Skipped experiments", ""])
        for r in missing:
            lines.append(f"- {r['model']} (seed {r['seed']}): {r.get('message', r['status'])}")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    (out / "aggregate.json").write_text(json.dumps(aggregates, indent=2) + "\n")
    if aggregates:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([r["model"] for r in aggregates], [r["f1_mean"] for r in aggregates],
               yerr=[r["f1_std"] for r in aggregates], capsize=4)
        ax.set(title="Held-out test F1 across seeds", ylabel="F1 — poor-quality class", ylim=(0, 1.05))
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(out / "model_comparison.png", dpi=150)
        plt.close(fig)
