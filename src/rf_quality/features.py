"""Numerical RF features shared by XGBoost and Isolation Forest.

All values are descriptive proxies, not scientifically verified quality labels.
No per-window z-scoring: amplitude differences remain observable.
"""

import numpy as np
from scipy.signal import welch

FEATURE_NAMES = (
    "mean_power", "power_std", "peak_power", "mean_amplitude", "amplitude_std",
    "amplitude_p99", "iq_mean_offset", "iq_variance_ratio", "zero_fraction",
    "peak_to_mean_power", "spectral_entropy", "spectral_flatness",
    "spectral_peak_ratio", "spectral_centroid", "spectral_spread",
    "phase_step_std", "amplitude_diff_std", "near_saturation_fraction",
    "estimated_snr_proxy_db", "noise_floor_db_relative", "peak_prominence_db",
    "frequency_step_std_cycles_per_sample",
)


def extract_features(iq_channels: np.ndarray) -> np.ndarray:
    """Convert a single (2, T) complex I/Q signal to a finite feature vector."""
    arr = np.asarray(iq_channels, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 2 or arr.shape[1] < 8:
        raise ValueError("Expected I/Q array of shape (2, at least 8 time points)")
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite I/Q samples are not accepted")
    z = arr[0] + 1j * arr[1]
    eps = 1e-12
    amp = np.abs(z)
    power = amp**2
    nperseg = min(256, len(z))
    _, psd = welch(z, nperseg=nperseg, return_onesided=False, detrend="constant")
    psd = np.maximum(psd.real, 0.0) + eps
    normalized = psd / psd.sum()
    entropy = -np.sum(normalized * np.log(normalized)) / np.log(len(psd))
    flatness = np.exp(np.mean(np.log(psd))) / psd.mean()
    frequencies = np.fft.fftfreq(len(psd))
    centroid = float(np.sum(np.abs(frequencies) * normalized))
    spread = float(np.sqrt(np.sum((np.abs(frequencies) - centroid)**2 * normalized)))
    phase_steps = np.angle(z[1:] * np.conj(z[:-1]))
    max_amplitude = amp.max()
    noise = float(np.median(psd))
    peak = float(psd.max())
    # Proxy only: the median PSD is NOT independently measured receiver noise.
    # Strong spectral peaks may correspond to interference, not desired transmitters.
    snr_proxy_db = 10 * np.log10(1 + max(0.0, peak - noise) / noise)
    noise_floor_db = 10 * np.log10(noise)
    peak_prominence_db = 10 * np.log10(peak / noise)
    values = [
        power.mean(), power.std(), power.max(), amp.mean(), amp.std(),
        np.percentile(amp, 99), np.abs(z.mean()),
        (np.var(arr[0]) + eps) / (np.var(arr[1]) + eps),
        np.mean(amp < 1e-9), power.max() / (power.mean() + eps),
        entropy, flatness, psd.max() / psd.mean(), centroid, spread,
        phase_steps.std(), np.diff(amp).std(),
        np.mean(amp >= 0.999 * max_amplitude) if max_amplitude > 0 else 1.0,
        snr_proxy_db, noise_floor_db, peak_prominence_db,
        phase_steps.std() / (2 * np.pi),
    ]
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Feature extraction produced invalid values")
    return result
