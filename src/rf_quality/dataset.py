"""Portable, no-pickle dataset format shared by all five model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, extract_features


@dataclass
class WindowDataset:
    signals: np.ndarray  # (N, 2, T), unnormalized I/Q
    features: np.ndarray  # (N, K)
    labels: np.ndarray  # 0=acceptable, 1=poor quality, -1=unknown
    station_ids: np.ndarray
    observation_ids: np.ndarray
    window_ids: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    origin: str = "unknown"
    timestamps: np.ndarray | None = None  # UTC seconds per window; required for temporal holdouts

    def validate(self) -> None:
        if self.signals.ndim != 3 or self.signals.shape[1] != 2:
            raise ValueError("signals must be [N, 2, T] (I/Q), with equal-length windows")
        n = len(self.signals)
        if self.features.shape != (n, len(self.feature_names)):
            raise ValueError("features shape mismatch")
        if any(len(x) != n for x in [self.labels, self.station_ids,
                                      self.observation_ids, self.window_ids]):
            raise ValueError("dataset row count mismatch")
        if self.timestamps is not None and (self.timestamps.shape != (n,) or not np.isfinite(self.timestamps).all()):
            raise ValueError("timestamps must be a finite [N] array of UTC seconds")
        if not np.isin(self.labels, [-1, 0, 1]).all():
            raise ValueError("labels must be -1, 0 or 1")
        if not np.isfinite(self.signals).all() or not np.isfinite(self.features).all():
            raise ValueError("nonfinite data; review missing-data handling")
        if len(set(map(str, self.window_ids))) != n:
            raise ValueError("window_ids must be unique")

    def save(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, signals=self.signals.astype(np.float32),
            features=self.features.astype(np.float32),
            labels=self.labels.astype(np.int8),
            station_ids=self.station_ids.astype(str),
            observation_ids=self.observation_ids.astype(str),
            window_ids=self.window_ids.astype(str),
            feature_names=np.asarray(self.feature_names, dtype=str),
            origin=np.asarray(self.origin, dtype=str),
            **({"timestamps": np.asarray(self.timestamps, dtype=np.float64)} if self.timestamps is not None else {}),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> WindowDataset:
        with np.load(path, allow_pickle=False) as z:
            result = cls(
                signals=z["signals"], features=z["features"],
                labels=z["labels"], station_ids=z["station_ids"],
                observation_ids=z["observation_ids"], window_ids=z["window_ids"],
                feature_names=tuple(map(str, z["feature_names"])),
                origin=str(z["origin"].item()),
                timestamps=z["timestamps"] if "timestamps" in z.files else None,
            )
        result.validate()
        return result


def assemble_windows(
    windows: list[np.ndarray], station_ids: list[str], observation_ids: list[str],
    window_ids: list[str], labels: list[int] | None = None, origin: str = "psws",
    feature_workers: int = 1,
    timestamps: list[float] | np.ndarray | None = None,
) -> WindowDataset:
    if not windows:
        raise ValueError("No readable windows. Check sample rate, channel paths and missing segments")
    signals = np.stack(windows).astype(np.float32)
    if feature_workers < 1:
        raise ValueError("feature_workers must be >=1")
    if feature_workers == 1:
        features = np.stack([extract_features(x) for x in signals]).astype(np.float32)
    else:
        try:
            from dask import compute, delayed
        except ImportError as exc:
            raise RuntimeError("Parallel feature extraction needs pip install -e '.[dask]'") from exc
        tasks = [delayed(extract_features)(x) for x in signals]
        features = np.stack(compute(*tasks, scheduler="threads", num_workers=feature_workers))
    return WindowDataset(
        signals=signals,
        features=features,
        labels=np.asarray(labels if labels is not None else [-1] * len(windows), dtype=np.int8),
        station_ids=np.asarray(station_ids, dtype=str),
        observation_ids=np.asarray(observation_ids, dtype=str),
        window_ids=np.asarray(window_ids, dtype=str), origin=origin,
        timestamps=np.asarray(timestamps, dtype=np.float64) if timestamps is not None else None,
    )
