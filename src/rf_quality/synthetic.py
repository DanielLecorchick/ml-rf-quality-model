"""Labeled SYNTHETIC RF examples solely for harness smoke tests.

Synthetic labels are NOT scientific ground truth for PSWS observations.
"""

import numpy as np

from .dataset import assemble_windows, WindowDataset


def generate_synthetic(*, stations: int = 8, per_station: int = 36,
                       length: int = 600, seed: int = 42) -> WindowDataset:
    if stations < 3 or per_station < 4 or length < 32:
        raise ValueError("Use >=3 stations, >=4 windows per station, >=32 samples")
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    station_ids, observations, ids, labels = [], [], [], []
    t = np.arange(length) / length
    for station in range(stations):
        for j in range(per_station):
            label = int(j % 3 == 0)
            f = 18 + rng.normal(0, 1)
            signal = np.exp(2j * np.pi * f * t)
            noise = 0.065 * (rng.standard_normal(length) + 1j * rng.standard_normal(length))
            z = signal + noise
            if label:
                mode = (j // 3) % 4
                if mode == 0:
                    z = z + 2.8 * (rng.standard_normal(length) + 1j * rng.standard_normal(length))
                elif mode == 1:
                    z[length // 4 : length // 2] = 0
                elif mode == 2:
                    z = z + 2.2 * np.exp(2j * np.pi * (f + 28) * t)
                else:
                    z = z * np.exp(1j * 28 * t * t)
            samples.append(np.stack([z.real, z.imag]).astype(np.float32))
            sid = f"S{station + 1:06d}"
            station_ids.append(sid)
            observations.append(f"synthetic:{sid}:day{j // 12}")
            ids.append(f"synthetic:{sid}:{j:04d}")
            labels.append(label)
    return assemble_windows(samples, station_ids, observations, ids, labels, origin="synthetic")
