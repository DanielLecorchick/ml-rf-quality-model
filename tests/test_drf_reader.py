import sys
import types
import numpy as np

from rf_quality.ingestion.digital_rf_reader import (
    find_observation_roots, iter_drf_windows, sample_rate,
)


class MockReader:
    def __init__(self, root):
        self.root = root
    def get_channels(self):
        return ["ch0"]
    def get_bounds(self, channel):
        return 0, 1200
    def get_properties(self, channel):
        return {"sample_rate_numerator": 10, "sample_rate_denominator": 1}
    def read_vector(self, index, count, channel):
        if index == 600:
            raise IOError("intentional dropout")
        return np.exp(2j * np.pi * np.arange(index, index + count) / 20)


def test_observation_discovery_and_read(monkeypatch, tmp_path):
    (tmp_path / "S000028" / "OBS2022-12-09T00-00" / "ch0").mkdir(parents=True)
    roots = find_observation_roots(tmp_path)
    assert len(roots) == 1
    mock = types.SimpleNamespace(DigitalRFReader=MockReader)
    monkeypatch.setitem(sys.modules, "digital_rf", mock)
    assert sample_rate(MockReader(""), "ch0") == 10
    windows = list(iter_drf_windows(roots[0], seconds=60, stride_seconds=60))
    assert len(windows) == 1
    assert windows[0].iq.shape == (2, 600)
    assert windows[0].station_id == "S000028"


def test_missing_drf_window_reported_to_audit_callback(monkeypatch, tmp_path):
    (tmp_path / "S000028" / "obs" / "ch0").mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "digital_rf", types.SimpleNamespace(DigitalRFReader=MockReader))
    events = []
    windows = list(iter_drf_windows(tmp_path / "S000028" / "obs", seconds=60,
                                    stride_seconds=60, on_gap=events.append))
    assert len(windows) == 1
    assert len(events) == 1
    assert events[0]["sample_index"] == 600
    assert events[0]["sample_count"] == 600
    assert events[0]["station_id"] == "S000028"
