"""Export annotation template and import human-reviewed binary quality labels."""

import csv
from pathlib import Path

import numpy as np

from .dataset import WindowDataset


def create_label_template(data: WindowDataset, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_id", "station_id", "observation_id", "label", "reviewer_notes"])
        for id_, station, obs, label in zip(
            data.window_ids, data.station_ids, data.observation_ids, data.labels
        ):
            writer.writerow([id_, station, obs, "" if label == -1 else label, ""])


def apply_labels(data: WindowDataset, csv_path: Path) -> WindowDataset:
    lookup = {str(id_): idx for idx, id_ in enumerate(data.window_ids)}
    values = np.array(data.labels, copy=True)
    seen: set[str] = set()
    with csv_path.open(newline="") as f:
        for line in csv.DictReader(f):
            key = str(line["window_id"])
            if key not in lookup:
                raise ValueError(f"Unknown window_id {key!r}")
            if key in seen:
                raise ValueError(f"Repeated window_id {key!r}")
            seen.add(key)
            label = str(line["label"]).strip()
            if label not in ("", "0", "1"):
                raise ValueError(f"Expected blank, 0 or 1 for {key}; got {label!r}")
            if label:
                values[lookup[key]] = int(label)
    result = WindowDataset(
        signals=data.signals, features=data.features, labels=values,
        station_ids=data.station_ids, observation_ids=data.observation_ids,
        window_ids=data.window_ids, feature_names=data.feature_names, origin=data.origin,
    )
    result.validate()
    return result
