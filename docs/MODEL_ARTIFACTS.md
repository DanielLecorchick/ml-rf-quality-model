# Trained model artifacts

`rf-quality train` saves the selected model (after validation-only hyperparameter and decision-threshold
selection) to `results/<run>/seed-<seed>/<model>/artifact/`.

Each directory contains:

- `manifest.json` (schema version, project version, timestamp, model name, seed, constructor parameters,
  SHA-256 checksum, class mapping, binary-decision threshold, probability-vs-anomaly score semantics,
  expected I/Q channel/window length, ordered RF feature names, training-only normalization statistics,
  dataset origin, training metadata, and installed package versions).
- `model.json` for XGBoost using XGBoost's native format;
  `model.pt` for the two PyTorch architectures (only `state_dict` tensor weights);
  or `model.joblib` for Isolation Forest and MultiROCKET (fitted estimator, plus training-fitted
  standard scaler for Isolation Forest).

Loading checks the schema, known model name, expected filename, and payload checksum. Neural models
are instantiated with stored architecture parameters and load tensors using `weights_only=True`.
Neural training means/stds are restored before inference; they are calculated **only on training**
windows, never on validation/test observations.

**Security:** `joblib.load` deserializes pickle and can execute code. **Never load bundles from
untrusted sources**, even if a checksum is present; a malicious actor could rewrite both payload
and manifest. The local demo API should bind only to loopback and load an explicitly specified
trusted bundle. Manifest hashes are for consistency, not cryptographic authentication.

Prediction compatibility is enforced by matching signal shape and ordered feature names. The scored
class is `1=poor` and higher scores indicate greater anomaly/poor-quality likelihood. Only XGBoost,
InceptionTime, and MultiROCKET's logistic-regression head produce probability-like outputs; these
have not been probability-calibrated. Isolation Forest and the autoencoder return raw anomaly scores.

For offline inference:

```bash
rf-quality predict-file \
  --model results/synthetic-smoke/seed-42/xgboost/artifact \
  --data data/processed/synthetic.npz \
  --out results/synthetic-smoke/scored.csv
```
