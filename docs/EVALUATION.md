# Reproducible evaluation workflow

## Splits

`rf-quality train` now creates disjoint **training / validation / test** splits.

- `--split-mode station` (default): all reviewed windows from a station belong to just one
  partition. Requires enough labeled stations with both classes in all three partitions.
- `--split-mode time`: within each station, use its earliest windows for training, middle windows
  for validation, and latest windows for test. Requires `timestamps` in the NPZ; they are derived
  from Digital RF sample indices and sample rate during `rf-quality build`. Use
  `--embargo-seconds` to omit near-boundary windows when observations are correlated/overlapping.

`--validation-fraction` and `--test-fraction` default to 0.2 each. Unknown (-1) labels are omitted.
The split JSON preserves window IDs, station IDs, class counts and seed. Even when using multiple
seeds, **never use the test metrics to select hyperparameters or thresholds**. Treat the plotted
seed-to-seed variation as uncertainty rather than an invitation to cherry-pick a split.

## Model search / imbalance / threshold selection

The optional JSON `--grid-file configs/example_grid.json` maps model names to configurations.
Each configuration is fitted on training windows and scored on validation windows; the best
validation objective (F1 by default) wins, with validation average precision as tie-breaker.
The validation set also chooses a decision threshold. Test windows are touched once for the
winner. Alternatives: `--threshold-metric balanced_accuracy` or `recall_at_precision` with
`--min-precision 0.8`. An unattainable precision requirement raises an error.

Imbalance mitigation: XGBoost uses `scale_pos_weight` computed on training labels; InceptionTime
uses weighted cross-entropy; MultiROCKET uses a balanced logistic regression estimator rather than
the RidgeClassifierCV default (to obtain meaningful continuous scores). Isolation Forest and the
convolutional autoencoder fit normal (acceptable) training windows only. Do not treat thresholds
or anomaly scores as calibrated event probabilities.

## Outputs

```text
results/<run>/
  summary.csv / summary.json
  aggregate.json / comparison.md / model_comparison.png
  seed-42/
    split.json
    xgboost/
      trials.json
      metrics.json
      test_predictions.csv
      confusion_matrix.png
      precision_recall.png
      feature_importance.png
      artifact/{manifest.json,model.json}
    autoencoder/... (model.pt)
    isolation_forest/... (model.joblib)
```

Synthetic data only verifies that the harness runs; it does not prove that the classifier can
recognize RF observation defects. Real scientific performance requires reviewed PSWS labels,
station coverage, and careful separation of true geophysical events from equipment faults.

## Example

```bash
rf-quality synthetic --out data/processed/synthetic.npz
rf-quality train --data data/processed/synthetic.npz \
  --models xgboost isolation_forest autoencoder --epochs 2 \
  --seeds 42 43 44 --out results/synthetic-validation

# More expensive search:
rf-quality train --data data/processed/psws_labeled.npz \
  --models all --split-mode station --grid-file configs/example_grid.json \
  --seeds 42 43 --out results/psws-experiments
```
