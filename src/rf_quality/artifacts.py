"""Versioned, locally trusted model bundles with reproducible preprocessing metadata.

The artifacts may include joblib pickles (IsolationForest, MultiROCKET). Only load
bundles you created and trust. SHA-256 detects accidental modification, not
malicious tampering if an attacker can rewrite the manifest too.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .dataset import WindowDataset
from .models.adapters import RFModel, make_model

SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "scikit-learn", "xgboost", "aeon", "torch", "tsai"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def save_model_bundle(
    model: RFModel, name: str, dataset: WindowDataset, directory: Path,
    *, seed: int, parameters: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save fitted model + feature schema + training-only normalization/threshold."""
    from .models.adapters import MODEL_PARAMS

    if name not in MODEL_PARAMS:
        raise ValueError(f"Unknown model: {name}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if name == "xgboost":
        filename = "model.json"
        model.model.save_model(str(directory / filename))
    elif name in ("isolation_forest", "multirocket"):
        import joblib
        filename = "model.joblib"
        state = {"estimator": model.model}
        if name == "isolation_forest":
            state["scaler"] = model.scaler
        joblib.dump(state, directory / filename)
    else:
        import torch
        filename = "model.pt"
        torch.save(model.model.cpu().state_dict(), directory / filename)
        model.model.to(model.device)
    stats = {}
    if hasattr(model, "means"):
        stats = {
            "iq_mean": np.asarray(model.means).reshape(2).tolist(),
            "iq_std": np.asarray(model.stds).reshape(2).tolist(),
            "scope": "training windows only; no per-window normalization",
        }
    threshold = float(model.threshold)
    if not np.isfinite(threshold):
        raise ValueError("Nonfinite decision threshold")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "project_version": __version__,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "seed": int(seed),
        "parameters": dict(parameters or {}),
        "model_file": filename,
        "sha256": _sha256(directory / filename),
        "decision_threshold": threshold,
        "score_is_probability": bool(model.probability),
        "class_mapping": {"0": "acceptable", "1": "poor-quality"},
        "channels": 2,
        "signal_length": int(dataset.signals.shape[2]),
        "feature_names": list(dataset.feature_names),
        "normalization": stats,
        "dataset_origin": dataset.origin,
        "library_versions": _versions(),
        "training": metadata or {},
        "joblib_requires_trusted_source": name in ("isolation_forest", "multirocket"),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return directory


def load_model_bundle(directory: Path) -> tuple[RFModel, dict[str, Any]]:
    """Load a *trusted* model bundle and verify its file checksum and schema."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported model-bundle schema version")
    from .models.adapters import MODEL_PARAMS

    name = manifest.get("name")
    if name not in MODEL_PARAMS:
        raise ValueError("Unknown model in manifest")
    filename = {"xgboost": "model.json", "isolation_forest": "model.joblib",
                "multirocket": "model.joblib", "inceptiontime": "model.pt",
                "autoencoder": "model.pt"}[name]
    if manifest.get("model_file") != filename:
        raise ValueError("Unexpected model filename")
    path = directory / filename
    if _sha256(path) != manifest.get("sha256"):
        raise ValueError("Model checksum does not match manifest")
    model = make_model(name, seed=int(manifest["seed"]), params=manifest["parameters"])
    if name == "xgboost":
        model.model.load_model(str(path))
    elif name in ("isolation_forest", "multirocket"):
        import joblib
        state = joblib.load(path)  # SECURITY: trusted local files only.
        model.model = state["estimator"]
        if name == "isolation_forest":
            model.scaler = state["scaler"]
    else:
        import torch
        model.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.model.to(model.device).eval()
        norm = manifest["normalization"]
        if len(norm.get("iq_mean", [])) != 2 or len(norm.get("iq_std", [])) != 2:
            raise ValueError("Invalid training normalization statistics")
        model.means = np.asarray(norm["iq_mean"], dtype=np.float32).reshape(1, 2, 1)
        model.stds = np.asarray(norm["iq_std"], dtype=np.float32).reshape(1, 2, 1)
    model.threshold = float(manifest["decision_threshold"])
    return model, manifest


def predict_bundle(model: RFModel, manifest: dict[str, Any], signals: np.ndarray,
                   features: np.ndarray, feature_names: tuple[str, ...] | list[str]):
    """Reject inputs incompatible with the fitted model before inference."""
    signals = np.asarray(signals, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if signals.ndim != 3 or signals.shape[1:] != (manifest["channels"], manifest["signal_length"]):
        raise ValueError(f"Expected signal shape (N, {manifest['channels']}, {manifest['signal_length']})")
    if tuple(feature_names) != tuple(manifest["feature_names"]):
        raise ValueError("Feature schema mismatch")
    if features.shape != (len(signals), len(feature_names)):
        raise ValueError("Feature array shape mismatch")
    if not np.isfinite(signals).all() or not np.isfinite(features).all():
        raise ValueError("Nonfinite input")
    return model.predict(signals, features)
