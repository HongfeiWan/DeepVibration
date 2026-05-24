"""Time-window exclusion cuts.

This module keeps two layers of behavior:

1. ``time_exclusion_mask``: a generic interval mask for any 1D time axis.
2. ``cut_time`` and helpers: the legacy CH0 burst-time cut used in the old
   analysis scripts.  It can either consume explicit bad intervals, or build
   them from the CH0 max-amplitude burst rate inside the May 20 -> June 10
   time window.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import numpy as np

CH0_BAND_BURST_LO = 1250.0
CH0_BAND_BURST_HI = 1500.0
CH0_TIME_BAND_BURST_RATE_THRESHOLD = 0.5  # count/min
CH0_TIME_EXCLUDE_BAD_INTERVALS_MPL: Tuple[Tuple[float, float], ...] = ()


def _normalize_intervals(
    bad_intervals: Iterable[Tuple[float, float]] | None,
) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    if bad_intervals is None:
        return intervals
    for start, stop in bad_intervals:
        a = float(start)
        b = float(stop)
        if np.isfinite(a) and np.isfinite(b) and b > a:
            intervals.append((a, b))
    return intervals


def _plot_window_mpl(year: int) -> Tuple[float, float]:
    """Legacy time window: May 20 to June 10 of the same year."""

    x_lo = mdates.date2num(datetime(year, 5, 20))
    x_hi = mdates.date2num(datetime(year, 6, 10))
    return x_lo, x_hi


def _merge_bad_bins_to_intervals(
    edges: np.ndarray,
    bad_bin: np.ndarray,
) -> List[Tuple[float, float]]:
    """Merge adjacent bad histogram bins into half-open intervals."""

    out: List[Tuple[float, float]] = []
    nbin = int(bad_bin.size)
    i = 0
    while i < nbin:
        if not bad_bin[i]:
            i += 1
            continue
        j = i
        while j + 1 < nbin and bad_bin[j + 1]:
            j += 1
        out.append((float(edges[i]), float(edges[j + 1])))
        i = j + 1
    return out


def time_exclusion_mask(
    time_values: np.ndarray,
    bad_intervals: Iterable[Tuple[float, float]] | None = None,
) -> np.ndarray:
    """Return ``True`` for events outside all bad intervals."""

    t = np.asarray(time_values, dtype=np.float64).reshape(-1)
    keep = np.ones(t.size, dtype=bool)
    for start, stop in _normalize_intervals(bad_intervals):
        keep &= ~((t >= start) & (t < stop))
    return keep


def compute_ch0_time_exclude_intervals_mpl(
    time_mpl: np.ndarray,
    max_ch0: np.ndarray,
    year: int,
    *,
    x_lo: Optional[float] = None,
    x_hi: Optional[float] = None,
    band_lo: float = CH0_BAND_BURST_LO,
    band_hi: float = CH0_BAND_BURST_HI,
    rate_threshold: float = CH0_TIME_BAND_BURST_RATE_THRESHOLD,
) -> List[Tuple[float, float]]:
    """Build bad intervals from CH0 burst rate in matplotlib-date units."""

    time_mpl = np.asarray(time_mpl, dtype=np.float64).reshape(-1)
    max_ch0 = np.asarray(max_ch0, dtype=np.float64).reshape(-1)
    n = min(time_mpl.size, max_ch0.size)
    time_mpl = time_mpl[:n]
    max_ch0 = max_ch0[:n]
    if x_lo is None or x_hi is None:
        x_lo, x_hi = _plot_window_mpl(year)
    n_bins = max(30, int((x_hi - x_lo) * 24))
    band = (max_ch0 >= band_lo) & (max_ch0 <= band_hi)
    time_band = time_mpl[band]
    counts, edges = np.histogram(time_band, bins=n_bins, range=(x_lo, x_hi))
    bin_width_days = float(edges[1] - edges[0])
    bin_width_min = bin_width_days * 24.0 * 60.0
    rate_per_min = counts.astype(np.float64) / bin_width_min
    bad = rate_per_min > float(rate_threshold)
    return _merge_bad_bins_to_intervals(edges, bad)


def build_ch0_time_exclude_intervals_global(
    time_mpl: np.ndarray,
    max_ch0: np.ndarray,
    pre_mask: Optional[np.ndarray] = None,
    *,
    rate_threshold: float = CH0_TIME_BAND_BURST_RATE_THRESHOLD,
    year: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """Build legacy bad intervals from already prepared event arrays."""

    t = np.asarray(time_mpl, dtype=np.float64).reshape(-1)
    x = np.asarray(max_ch0, dtype=np.float64).reshape(-1)
    n = min(t.size, x.size)
    if n == 0:
        return []
    t = t[:n]
    x = x[:n]

    if pre_mask is not None:
        m = np.asarray(pre_mask, dtype=bool).reshape(-1)[:n]
        t = t[m]
        x = x[m]
    if t.size == 0:
        return []

    resolved_year = int(mdates.num2date(float(t[0])).year if year is None else int(year))
    return compute_ch0_time_exclude_intervals_mpl(
        t,
        x,
        resolved_year,
        rate_threshold=rate_threshold,
    )


def _bad_intervals_total_days(intervals: Sequence[Tuple[float, float]]) -> float:
    """Total length of intervals in days."""

    total_days = 0.0
    for start, stop in intervals:
        a = float(start)
        b = float(stop)
        if np.isfinite(a) and np.isfinite(b) and b > a:
            total_days += b - a
    return float(total_days)


def cut_time(
    time_mpl: np.ndarray,
    bad_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    max_ch0: Optional[np.ndarray] = None,
    pre_mask: Optional[np.ndarray] = None,
    rate_threshold: float = CH0_TIME_BAND_BURST_RATE_THRESHOLD,
    year: Optional[int] = None,
    return_intervals: bool = False,
) -> np.ndarray | Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Legacy time cut: ``True`` means keep, ``False`` means reject."""

    if bad_intervals is not None:
        intervals = _normalize_intervals(bad_intervals)
    elif max_ch0 is not None and pre_mask is not None:
        intervals = build_ch0_time_exclude_intervals_global(
            time_mpl,
            max_ch0,
            pre_mask=pre_mask,
            rate_threshold=rate_threshold,
            year=year,
        )
    else:
        intervals = _normalize_intervals(CH0_TIME_EXCLUDE_BAD_INTERVALS_MPL)

    keep = time_exclusion_mask(time_mpl, intervals)
    return (keep, intervals) if return_intervals else keep


cut_time.bad_intervals_total_days = _bad_intervals_total_days  # type: ignore[attr-defined]


__all__ = [
    "CH0_BAND_BURST_LO",
    "CH0_BAND_BURST_HI",
    "CH0_TIME_BAND_BURST_RATE_THRESHOLD",
    "CH0_TIME_EXCLUDE_BAD_INTERVALS_MPL",
    "build_ch0_time_exclude_intervals_global",
    "compute_ch0_time_exclude_intervals_mpl",
    "cut_time",
    "time_exclusion_mask",
]
