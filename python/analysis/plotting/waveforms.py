"""Waveform plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.features import time_axis


def plot_waveform(
    waveform: np.ndarray,
    *,
    sampling_interval_ns: float = 4.0,
    unit: str = "us",
    ax=None,
    label: str | None = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.5))
    x = time_axis(np.asarray(waveform).size, sampling_interval_ns=sampling_interval_ns, unit=unit)
    ax.plot(x, waveform, linewidth=1.0, label=label)
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("Amplitude (ADC counts)")
    if label:
        ax.legend()
    return ax


__all__ = ["plot_waveform"]
