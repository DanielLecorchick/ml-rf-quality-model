"""Stream time windows out of extracted Grape Digital RF/HDF5 observations.

The optional digital-rf distribution supplies the reader implementation.
A directory *above* ch0/ is the observation root; do not point at ch0 itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Callable, Iterator

import numpy as np

LOG = logging.getLogger(__name__)


@dataclass
class DRFWindow:
    iq: np.ndarray  # float32 (2, T) real/imag
    station_id: str
    observation_id: str
    window_id: str
    sample_index: int
    sample_rate_hz: float
    timestamp_unix: float


def find_observation_roots(folder: Path, channel: str = "ch0") -> list[Path]:
    """Find Digital RF reader roots using the directory structure documented by HamSCI."""
    folder = folder.expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(folder)
    candidates = [folder] if (folder / channel).is_dir() else []
    candidates.extend(p.parent for p in folder.rglob(channel) if p.is_dir())
    return sorted(set(candidates))


def sample_rate(reader, channel: str) -> float:
    properties = reader.get_properties(channel)
    num = properties.get("sample_rate_numerator")
    den = properties.get("sample_rate_denominator")
    if num is not None and den:
        return float(num) / float(den)
    rate = properties.get("samples_per_second")
    if rate is not None:
        return float(rate)
    raise ValueError(f"Unable to infer sample rate from channel {channel} properties: {properties}")


def iter_drf_windows(
    root: Path,
    *,
    channel: str = "ch0",
    seconds: float = 60,
    stride_seconds: float = 60,
    station_id: str | None = None,
    max_windows: int | None = None,
    on_gap: Callable[[dict[str, object]], None] | None = None,
) -> Iterator[DRFWindow]:
    try:
        import digital_rf as drf
    except ImportError as exc:
        raise RuntimeError("Digital RF not installed. Run pip install -e '.[drf]'") from exc
    reader = drf.DigitalRFReader(str(root))
    channels = reader.get_channels()
    if channel not in channels:
        raise ValueError(f"Channel {channel!r} not in {channels!r}")
    rate = sample_rate(reader, channel)
    window_len = round(seconds * rate)
    stride = round(stride_seconds * rate)
    if window_len < 8 or stride < 1:
        raise ValueError("Window or stride has too few samples")
    start, stop = reader.get_bounds(channel)
    resolved_station = station_id or next(
        (match.group(0) for part in root.parts if (match := re.search(r"S\d{6,}", part))),
        f"unknown_{root.name}",
    )
    observation_id = f"{resolved_station}:{root.name}"
    emitted = 0
    for idx in range(int(start), int(stop) - window_len + 1, stride):
        try:
            samples = np.asarray(reader.read_vector(idx, window_len, channel))
            if samples.shape != (window_len,) or not np.isfinite(samples).all():
                raise ValueError("Unexpected shape or non-finite data")
        except (OSError, IOError, RuntimeError, ValueError, KeyError) as exc:
            LOG.warning("Skipping unreadable/gap-containing window in %s at %d: %s", root, idx, exc)
            if on_gap is not None:
                on_gap({"station_id": resolved_station, "observation_id": observation_id,
                        "sample_index": int(idx), "sample_count": int(window_len),
                        "sample_rate_hz": float(rate), "timestamp_unix": float(idx / rate),
                        "reason": str(exc)[:512]})
            continue
        iq = np.stack([samples.real, samples.imag]).astype(np.float32)
        yield DRFWindow(
            iq=iq, station_id=resolved_station, observation_id=observation_id,
            window_id=f"{observation_id}:{idx}", sample_index=idx, sample_rate_hz=rate,
            timestamp_unix=idx / rate,
        )
        emitted += 1
        if max_windows is not None and emitted >= max_windows:
            break
