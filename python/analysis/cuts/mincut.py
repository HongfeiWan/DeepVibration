"""Minimum-amplitude cut helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def mincut_mask(
    ch0_min: np.ndarray,
    ch1_min: np.ndarray,
    fit_mask: np.ndarray | None = None,
    *,
    n_sigma: float = 3.0,
    min_fit_events: int = 10,
    return_stats: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float]]:
    """Fit CH0/CH1 minimum distributions and keep events inside ``n_sigma``."""

    x = np.asarray(ch0_min, dtype=np.float64).reshape(-1)
    y = np.asarray(ch1_min, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    if fit_mask is None:
        fit = np.ones(n, dtype=bool)
    else:
        fit = np.asarray(fit_mask, dtype=bool).reshape(-1)[:n]

    mask = np.ones(n, dtype=bool)
    stats: Dict[str, float] = {"fit_events": float(np.count_nonzero(fit))}
    for label, arr in (("ch0", x), ("ch1", y)):
        sample = arr[fit]
        if sample.size < int(min_fit_events):
            stats[f"{label}_used"] = 0.0
            continue
        mu = float(sample.mean())
        sigma = float(sample.std(ddof=1))
        stats[f"{label}_mu"] = mu
        stats[f"{label}_sigma"] = sigma
        stats[f"{label}_used"] = 1.0
        if sigma > 0.0:
            mask &= np.abs(arr - mu) <= float(n_sigma) * sigma
    return (mask, stats) if return_stats else mask


__all__ = ["mincut_mask"]
