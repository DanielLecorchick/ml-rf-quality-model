"""Adapters for established model implementations.

No upstream model sources are pasted here. XGBoost, sklearn, aeon, and tsai
are imported from their official Python packages. Our autoencoder is a small
original PyTorch baseline.

Binary semantics: 0 = acceptable, 1 = poor-quality; larger score = worse.
For anomaly detectors, score is NOT a calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelPrediction:
    score: np.ndarray  # large score -> more likely poor quality
    predicted: np.ndarray
    is_probability: bool


class RFModel:
    probability = False
    threshold = 0.5

    def fit(self, signals: np.ndarray, features: np.ndarray, labels: np.ndarray) -> None:
        raise NotImplementedError

    def score(self, signals: np.ndarray, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, signals: np.ndarray, features: np.ndarray) -> ModelPrediction:
        score = np.asarray(self.score(signals, features), dtype=float).reshape(-1)
        cutoff = self.threshold
        return ModelPrediction(score, (score >= cutoff).astype(np.int8), self.probability)


class XGBoostRF(RFModel):
    probability = True

    def __init__(self, seed: int = 42, n_estimators: int = 150, max_depth: int = 4,
                 learning_rate: float = 0.06, class_balance: bool = True):
        self.class_balance = class_balance
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("Missing XGBoost: pip install -e '.[xgboost]'") from exc
        self.model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=0.85, colsample_bytree=0.85, objective="binary:logistic",
            eval_metric="logloss", random_state=seed, n_jobs=2,
        )

    def fit(self, signals, features, labels):
        if self.class_balance:
            labels = np.asarray(labels)
            pos = np.count_nonzero(labels == 1)
            neg = np.count_nonzero(labels == 0)
            if pos and neg:
                self.model.set_params(scale_pos_weight=neg / pos)
        self.model.fit(features, labels)

    def score(self, signals, features):
        return self.model.predict_proba(features)[:, 1]


class MultiRocketRF(RFModel):
    probability = True

    def __init__(self, seed: int = 42, kernels: int = 2000, class_balance: bool = True):
        try:
            from aeon.classification.convolution_based import MultiRocketClassifier
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise RuntimeError("Missing aeon: pip install -e '.[rocket]'") from exc
        self.model = MultiRocketClassifier(
            n_kernels=kernels, n_jobs=2, random_state=seed,
            estimator=LogisticRegression(
                max_iter=2000, class_weight="balanced" if class_balance else None,
                random_state=seed,
            ),
        )

    def fit(self, signals, features, labels):
        self.model.fit(signals, labels)

    def score(self, signals, features):
        return self.model.predict_proba(signals)[:, 1]


class IsolationForestRF(RFModel):
    """Fit on label=0 train examples; threshold = 95th percentile of train normal scores."""

    def __init__(self, seed: int = 42, n_estimators: int = 200):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=2)

    def fit(self, signals, features, labels):
        normal = features[np.asarray(labels) == 0]
        if len(normal) < 3:
            raise ValueError("Isolation Forest requires at least 3 labeled acceptable train windows")
        normalized = self.scaler.fit_transform(normal)
        self.model.fit(normalized)
        raw = -self.model.decision_function(normalized)
        self.threshold = float(np.quantile(raw, .95))

    def score(self, signals, features):
        return -self.model.decision_function(self.scaler.transform(features))


def _torch():
    try:
        import torch
        from torch import nn
        return torch, nn
    except ImportError as exc:
        raise RuntimeError("Missing PyTorch: pip install -e '.[autoencoder]'") from exc


class _NeuralBase(RFModel):
    def __init__(self, seed: int = 42, epochs: int = 8, batch_size: int = 32):
        torch, _ = _torch()
        torch.manual_seed(seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _normalize_fit(self, signals):
        # Global TRAINING statistics, never independently normalize each observation.
        self.means = signals.mean(axis=(0, 2), keepdims=True)
        self.stds = np.maximum(signals.std(axis=(0, 2), keepdims=True), 1e-5)

    def _tensor(self, signals):
        torch, _ = _torch()
        x = (signals - self.means) / self.stds
        return torch.tensor(x, dtype=torch.float32)


class InceptionTimeRF(_NeuralBase):
    """Original InceptionTime architecture from the existing tsai PyTorch library."""

    probability = True

    def __init__(self, seed: int = 42, epochs: int = 8, batch_size: int = 32,
                 filters: int = 16, class_balance: bool = True):
        super().__init__(seed, epochs, batch_size)
        self.class_balance = class_balance
        self.filters = filters
        try:
            from tsai.models.InceptionTime import InceptionTime
        except ImportError as exc:
            raise RuntimeError("Missing tsai: pip install -e '.[inception]'") from exc
        self.model = InceptionTime(c_in=2, c_out=2, nf=filters).to(self.device)

    def fit(self, signals, features, labels):
        torch, nn = _torch()
        self._normalize_fit(signals)
        weights = None
        if self.class_balance:
            counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
            weights = torch.as_tensor(
                len(labels) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=self.device
            )
        X = self._tensor(signals)
        y = torch.tensor(labels, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y), batch_size=self.batch_size, shuffle=True
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(self.model(xb), yb, weight=weights)
                loss.backward()
                optimizer.step()

    def score(self, signals, features):
        torch, _ = _torch()
        self.model.eval()
        chunks = []
        with torch.no_grad():
            for xb in self._tensor(signals).split(self.batch_size):
                probs = torch.softmax(self.model(xb.to(self.device)), dim=1)[:, 1]
                chunks.append(probs.cpu().numpy())
        return np.concatenate(chunks)


class ConvAutoencoderRF(_NeuralBase):
    """Small convolutional autoencoder, trained on known-normal windows only."""

    def __init__(self, seed: int = 42, epochs: int = 8, batch_size: int = 32):
        super().__init__(seed, epochs, batch_size)
        torch, nn = _torch()
        self.model = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=7, padding=3),
        ).to(self.device)

    def _errors(self, signals):
        torch, nn = _torch()
        self.model.eval()
        chunks = []
        with torch.no_grad():
            for xb in self._tensor(signals).split(self.batch_size):
                xb = xb.to(self.device)
                reconstructed = self.model(xb)
                # Handle odd-length signals without changing the original sequence length.
                reconstructed = nn.functional.pad(reconstructed, (0, max(0, xb.shape[-1] - reconstructed.shape[-1])))
                reconstructed = reconstructed[..., :xb.shape[-1]]
                chunks.append(((reconstructed - xb) ** 2).mean(dim=(1, 2)).cpu().numpy())
        return np.concatenate(chunks)

    def fit(self, signals, features, labels):
        torch, nn = _torch()
        normal = signals[np.asarray(labels) == 0]
        if len(normal) < 3:
            raise ValueError("Autoencoder requires at least 3 labeled acceptable train windows")
        self._normalize_fit(normal)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self._tensor(normal)),
            batch_size=self.batch_size, shuffle=True,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.model.train()
        for _ in range(self.epochs):
            for (xb,) in loader:
                xb = xb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = self.model(xb)
                reconstruction = nn.functional.pad(reconstruction, (0, max(0, xb.shape[-1] - reconstruction.shape[-1])))
                reconstruction = reconstruction[..., :xb.shape[-1]]
                loss = nn.functional.mse_loss(reconstruction, xb)
                loss.backward()
                optimizer.step()
        self.threshold = float(np.quantile(self._errors(normal), .95))

    def score(self, signals, features):
        return self._errors(signals)


AVAILABLE_MODELS = ("xgboost", "inceptiontime", "multirocket", "autoencoder", "isolation_forest")


MODEL_PARAMS = {
    "xgboost": {"n_estimators", "max_depth", "learning_rate", "class_balance"},
    "multirocket": {"kernels", "class_balance"},
    "isolation_forest": {"n_estimators"},
    "inceptiontime": {"epochs", "batch_size", "filters", "class_balance"},
    "autoencoder": {"epochs", "batch_size"},
}


def make_model(name: str, *, seed: int = 42, epochs: int = 8,
               rocket_kernels: int = 2000, params: dict | None = None) -> RFModel:
    if name not in MODEL_PARAMS:
        raise ValueError(f"Unknown model {name!r}; choose from {AVAILABLE_MODELS}")
    kwargs = dict(params or {})
    unknown = set(kwargs) - MODEL_PARAMS[name]
    if unknown:
        raise ValueError(f"Unsupported {name} hyperparameters: {sorted(unknown)}")
    if name == "xgboost":
        return XGBoostRF(seed=seed, **kwargs)
    if name == "multirocket":
        kwargs.setdefault("kernels", rocket_kernels)
        return MultiRocketRF(seed=seed, **kwargs)
    if name == "isolation_forest":
        return IsolationForestRF(seed=seed, **kwargs)
    kwargs.setdefault("epochs", epochs)
    if name == "inceptiontime":
        return InceptionTimeRF(seed=seed, **kwargs)
    return ConvAutoencoderRF(seed=seed, **kwargs)

