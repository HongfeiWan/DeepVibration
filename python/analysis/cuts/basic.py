"""Basic event-classification cuts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np


@dataclass
class CutResult:
    """Mask plus bookkeeping for a cut sequence."""

    mask: np.ndarray
    stats: Dict[str, float] = field(default_factory=dict)
    steps: Dict[str, np.ndarray] = field(default_factory=dict)


def _as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _common_length(*arrays: np.ndarray) -> int:
    if not arrays:
        return 0
    return min(np.asarray(arr).size for arr in arrays)


def summarize_mask(mask: np.ndarray) -> Dict[str, float]:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    total = int(m.size)
    passed = int(np.count_nonzero(m))
    return {
        "total": float(total),
        "passed": float(passed),
        "failed": float(total - passed),
        "pass_fraction": float(passed / total) if total else 0.0,
    }


def rt_mask(max_ch5: np.ndarray, threshold: float = 6000.0) -> np.ndarray:
    """Random-trigger mask: ``max(CH5) > threshold``."""

    return _as_float_array(max_ch5) > float(threshold)


def inhibit_mask(ch0_min: np.ndarray, zero_value: float = 0.0) -> np.ndarray:
    """Inhibit mask: ``min(CH0) == zero_value``."""

    return _as_float_array(ch0_min) == float(zero_value)


def saturation_mask(
    max_ch0: np.ndarray,
    max_ch1: np.ndarray | None = None,
    *,
    max_adc: float = 16382.0,
) -> np.ndarray:
    """Reject saturated events; if CH1 is supplied, require both channels."""

    ch0 = _as_float_array(max_ch0)
    if max_ch1 is None:
        return ch0 <= float(max_adc)
    ch1 = _as_float_array(max_ch1)
    n = _common_length(ch0, ch1)
    return (ch0[:n] <= float(max_adc)) & (ch1[:n] <= float(max_adc))


def _single_sigma_band_mask(
    values: np.ndarray,
    fit_mask: np.ndarray,
    *,
    n_sigma: float,
    min_fit_events: int,
    prefix: str,
) -> tuple[np.ndarray, Dict[str, float]]:
    arr = _as_float_array(values)
    fit = np.asarray(fit_mask, dtype=bool).reshape(-1)
    n = _common_length(arr, fit)
    arr = arr[:n]
    fit = fit[:n]
    finite_fit = fit & np.isfinite(arr)
    stats: Dict[str, float] = {f"{prefix}_fit_events": float(np.count_nonzero(finite_fit))}
    keep = np.ones(n, dtype=bool)
    if np.count_nonzero(finite_fit) < int(min_fit_events):
        stats[f"{prefix}_used"] = 0.0
        return keep, stats

    sample = arr[finite_fit]
    mu = float(sample.mean())
    sigma = float(sample.std(ddof=1))
    stats.update({
        f"{prefix}_used": 1.0,
        f"{prefix}_mu": mu,
        f"{prefix}_sigma": sigma,
    })
    if sigma <= 0.0:
        return keep, stats
    keep = np.isfinite(arr) & (np.abs(arr - mu) <= float(n_sigma) * sigma)
    return keep, stats


def pedestal_3sigma_mask(
    ch0_ped_mean: np.ndarray,
    ch1_ped_mean: np.ndarray,
    reference_mask: np.ndarray,
    *,
    n_sigma: float = 3.0,
    min_fit_events: int = 10,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, float]]:
    """Keep events whose CH0/CH1 pedestal means stay inside the reference 3 sigma band."""

    ch0 = _as_float_array(ch0_ped_mean)
    ch1 = _as_float_array(ch1_ped_mean)
    ref = np.asarray(reference_mask, dtype=bool).reshape(-1)
    n = _common_length(ch0, ch1, ref)
    ch0_keep, ch0_stats = _single_sigma_band_mask(
        ch0[:n],
        ref[:n],
        n_sigma=n_sigma,
        min_fit_events=min_fit_events,
        prefix="ch0_pedestal",
    )
    ch1_keep, ch1_stats = _single_sigma_band_mask(
        ch1[:n],
        ref[:n],
        n_sigma=n_sigma,
        min_fit_events=min_fit_events,
        prefix="ch1_pedestal",
    )
    keep = ch0_keep & ch1_keep
    stats: Dict[str, float] = {}
    stats.update(ch0_stats)
    stats.update(ch1_stats)
    stats.update({f"combined_{k}": v for k, v in summarize_mask(keep).items()})
    return (keep, stats) if return_stats else keep


def combine_steps(steps: Mapping[str, np.ndarray]) -> CutResult:
    """Combine named boolean masks and keep per-step summaries."""

    if not steps:
        return CutResult(mask=np.array([], dtype=bool))
    n = min(np.asarray(mask).size for mask in steps.values())
    combined = np.ones(n, dtype=bool)
    normalized: Dict[str, np.ndarray] = {}
    stats: Dict[str, float] = {}
    for name, mask in steps.items():
        m = np.asarray(mask, dtype=bool).reshape(-1)[:n]
        normalized[name] = m
        combined &= m
        stats[f"{name}_passed"] = float(np.count_nonzero(m))
    stats.update({f"combined_{k}": v for k, v in summarize_mask(combined).items()})
    return CutResult(mask=combined, stats=stats, steps=normalized)


__all__ = [
    "CutResult",
    "combine_steps",
    "inhibit_mask",
    "pedestal_3sigma_mask",
    "rt_mask",
    "saturation_mask",
    "summarize_mask",
]
