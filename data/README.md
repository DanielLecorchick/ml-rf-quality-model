# Data conventions

- `raw/`: downloads from PSWS API, with adjacent SHA-256 JSON manifests; ignored by Git
- `extracted/`: extracted Digital RF observations; ignored by Git
- `processed/`: NPZ window datasets; ignored by Git
- `labels/`: human-reviewed quality labels; ignored by Git by default

**Labels:** -1 unknown, 0 acceptable, 1 poor quality. Synthetic labels are not PSWS truth.
The model's larger score always means "more likely poor-quality" or "more anomalous".
