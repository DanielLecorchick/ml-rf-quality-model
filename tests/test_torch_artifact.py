import numpy as np
import pytest

from rf_quality.artifacts import load_model_bundle, predict_bundle, save_model_bundle
from rf_quality.models.adapters import ConvAutoencoderRF
from rf_quality.synthetic import generate_synthetic


def test_autoencoder_roundtrip_training_stats_and_scores(tmp_path):
    pytest.importorskip("torch")
    ds = generate_synthetic(stations=4, per_station=8, length=64)
    model = ConvAutoencoderRF(seed=2, epochs=1)
    model.fit(ds.signals, ds.features, ds.labels)
    first = model.score(ds.signals[:4], ds.features[:4])
    saved = save_model_bundle(model, "autoencoder", ds, tmp_path / "ae", seed=2,
                              parameters={"epochs": 1})
    loaded, manifest = load_model_bundle(saved)
    again = predict_bundle(loaded, manifest, ds.signals[:4], ds.features[:4], ds.feature_names)
    np.testing.assert_allclose(first, again.score, rtol=1e-6, atol=1e-7)
    assert len(manifest["normalization"]["iq_mean"]) == 2
    assert len(manifest["normalization"]["iq_std"]) == 2
    assert manifest["score_is_probability"] is False
