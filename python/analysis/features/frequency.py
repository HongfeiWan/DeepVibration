"""Frequency-domain waveform features."""

from __future__ import annotations

import numpy as np


def _prepare_waveform(waveform: np.ndarray, sampling_interval_ns: float) -> np.ndarray:
    wf = np.asarray(waveform, dtype=np.float64)
    n_120 = int(round(120.0 * 1000.0 / sampling_interval_ns))
    wf = wf[: min(n_120, wf.size)]
    if wf.size == 0:
        return wf
    wf = wf - np.mean(wf)
    if wf.size > 1:
        wf = wf * np.hanning(wf.size)
    return wf


def compute_highfreq_energy_ratio(
    waveform: np.ndarray,
    *,
    sampling_interval_ns: float = 4.0,
    cutoff_mhz: float = 0.2,
) -> float:
    wf = _prepare_waveform(waveform, sampling_interval_ns)
    if wf.size == 0:
        return 0.0
    freq = np.fft.rfftfreq(wf.size, d=sampling_interval_ns * 1e-9)
    power = np.abs(np.fft.rfft(wf)) ** 2
    non_dc = freq > 0.0
    total = float(np.sum(power[non_dc]))
    if total <= 0.0:
        return 0.0
    return float(np.sum(power[freq >= cutoff_mhz * 1e6]) / total)


def compute_spectral_centroid_mhz(
    waveform: np.ndarray,
    *,
    sampling_interval_ns: float = 4.0,
) -> float:
    wf = _prepare_waveform(waveform, sampling_interval_ns)
    if wf.size == 0:
        return 0.0
    freq = np.fft.rfftfreq(wf.size, d=sampling_interval_ns * 1e-9)
    power = np.abs(np.fft.rfft(wf)) ** 2
    non_dc = freq > 0.0
    total = float(np.sum(power[non_dc]))
    if total <= 0.0:
        return 0.0
    return float(np.sum(freq[non_dc] * power[non_dc]) / total) * 1e-6


__all__ = ["compute_highfreq_energy_ratio", "compute_spectral_centroid_mhz"]
