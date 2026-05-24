"""PN correlation cut helpers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _fit_line(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_fit_events: int,
) -> Optional[Tuple[float, float, float]]:
    if x.size < min_fit_events:
        return None
    slope, intercept = np.polyfit(x, y, deg=1)
    resid = y - (slope * x + intercept)
    sigma = float(resid.std(ddof=1))
    if sigma <= 0.0:
        return None
    return float(slope), float(intercept), sigma


def pncut_mask(
    max_ch0: np.ndarray,
    max_ch1: np.ndarray,
    base_mask: np.ndarray | None = None,
    *,
    fit_ch0_min: float = 3000.0,
    fit_ch0_max: float = 12000.0,
    n_sigma: float = 0.8,
    min_fit_events: int = 10,
    return_stats: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float]]:
    """Keep events in the CH0/CH1 main linear correlation band."""

    x = np.asarray(max_ch0, dtype=np.float64).reshape(-1)
    y = np.asarray(max_ch1, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    if base_mask is None:
        base = np.ones(n, dtype=bool)
    else:
        base = np.asarray(base_mask, dtype=bool).reshape(-1)[:n]

    fit_mask = base & (x > float(fit_ch0_min)) & (x < float(fit_ch0_max))
    fit = _fit_line(x[fit_mask], y[fit_mask], min_fit_events=int(min_fit_events))
    stats: Dict[str, float] = {"fit_events": float(np.count_nonzero(fit_mask))}
    if fit is None:
        out = base.copy()
        stats["fit_ok"] = 0.0
        return (out, stats) if return_stats else out

    slope, intercept, sigma = fit
    resid = y - (slope * x + intercept)
    out = np.abs(resid) <= float(n_sigma) * sigma
    stats.update({"fit_ok": 1.0, "slope": slope, "intercept": intercept, "sigma": sigma})
    return (out, stats) if return_stats else out


__all__ = ["pncut_mask"]
