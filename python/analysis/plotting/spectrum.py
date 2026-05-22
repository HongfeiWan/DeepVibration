"""Spectrum plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrum(
    energy: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    bins: int | np.ndarray = 200,
    ax=None,
    label: str | None = None,
):
    values = np.asarray(energy, dtype=np.float64).reshape(-1)
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        values = values[: m.size][m]
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(values, bins=bins, histtype="step", linewidth=1.5, label=label)
    ax.set_xlabel("Energy")
    ax.set_ylabel("Counts")
    if label:
        ax.legend()
    return ax


__all__ = ["plot_spectrum"]
