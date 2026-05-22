"""Signal-processing transforms shared by FFT/Lomb/Hilbert/Wavelet scripts."""

from __future__ import annotations

import numpy as np


def fft_power_spectrum(waveform: np.ndarray, *, sampling_interval_ns: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    wf = np.asarray(waveform, dtype=np.float64)
    if wf.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    wf = wf - np.mean(wf)
    if wf.size > 1:
        wf = wf * np.hanning(wf.size)
    freq_hz = np.fft.rfftfreq(wf.size, d=sampling_interval_ns * 1e-9)
    power = np.abs(np.fft.rfft(wf)) ** 2
    return freq_hz, power


def lomb_scargle_power(
    time_s: np.ndarray,
    waveform: np.ndarray,
    freq_hz: np.ndarray,
) -> np.ndarray:
    try:
        from scipy.signal import lombscargle
    except Exception as exc:
        raise ImportError("Install scipy to run Lomb-Scargle analysis") from exc

    t = np.asarray(time_s, dtype=np.float64)
    y = np.asarray(waveform, dtype=np.float64)
    f = np.asarray(freq_hz, dtype=np.float64)
    if t.shape != y.shape:
        raise ValueError("time_s and waveform must have the same shape")
    return lombscargle(t, y - np.mean(y), 2.0 * np.pi * f, normalize=True)


def hilbert_envelope(waveform: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import hilbert
    except Exception as exc:
        raise ImportError("Install scipy to run Hilbert analysis") from exc

    return np.abs(hilbert(np.asarray(waveform, dtype=np.float64)))


def continuous_wavelet_power(
    waveform: np.ndarray,
    *,
    scales: np.ndarray,
    wavelet: str = "cmor3-3",
    sampling_interval_ns: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pywt
    except Exception as exc:
        raise ImportError("Install the 'signal' extra to run wavelet analysis") from exc

    coeffs, freqs = pywt.cwt(
        np.asarray(waveform, dtype=np.float64),
        scales,
        wavelet,
        sampling_period=sampling_interval_ns * 1e-9,
    )
    return freqs, np.abs(coeffs) ** 2


__all__ = [
    "continuous_wavelet_power",
    "fft_power_spectrum",
    "hilbert_envelope",
    "lomb_scargle_power",
]
