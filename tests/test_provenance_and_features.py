import json
import numpy as np

from rf_quality.features import FEATURE_NAMES, extract_features
from rf_quality.provenance import sha256_file, write_run_provenance


def test_additional_rf_proxy_features_are_finite():
    t = np.linspace(0, 1, 256, endpoint=False)
    iq = np.stack([np.cos(2 * np.pi * 12 * t), np.sin(2 * np.pi * 12 * t)])
    vals = extract_features(iq)
    assert len(FEATURE_NAMES) == 22
    assert np.isfinite(vals).all()
    assert vals[FEATURE_NAMES.index("estimated_snr_proxy_db")] >= 0
    assert vals[FEATURE_NAMES.index("peak_prominence_db")] >= 0


def test_run_provenance_records_input_hash(tmp_path):
    source = tmp_path / "input.npz"
    source.write_bytes(b"dataset")
    dest = write_run_provenance(source, tmp_path / "results", arguments={"seeds": [42,43]})
    manifest = json.loads(dest.read_text())
    assert manifest["dataset_sha256"] == sha256_file(source)
    assert manifest["arguments"]["seeds"] == [42,43]
