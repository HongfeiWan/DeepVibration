"""Time-window exclusion cuts."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def time_exclusion_mask(
    time_values: np.ndarray,
    bad_intervals: Iterable[Tuple[float, float]],
) -> np.ndarray:
    """Return True for events outside all bad intervals."""

    t = np.asarray(time_values, dtype=np.float64).reshape(-1)
    keep = np.ones(t.size, dtype=bool)
    for start, stop in bad_intervals:
        start_f = float(start)
        stop_f = float(stop)
        if np.isfinite(start_f) and np.isfinite(stop_f) and stop_f > start_f:
            keep &= ~((t >= start_f) & (t < stop_f))
    return keep


__all__ = ["time_exclusion_mask"]
