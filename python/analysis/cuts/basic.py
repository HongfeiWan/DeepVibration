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


def physical_mask(
    max_ch5: np.ndarray,
    ch0_min: np.ndarray,
    *,
    rt_threshold: float = 6000.0,
    inhibit_zero_value: float = 0.0,
) -> np.ndarray:
    """Physical self-trigger candidates: neither RT nor inhibit."""

    n = _common_length(max_ch5, ch0_min)
    rt = rt_mask(max_ch5[:n], rt_threshold)
    inh = inhibit_mask(ch0_min[:n], inhibit_zero_value)
    return (~rt) & (~inh)


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
    "physical_mask",
    "rt_mask",
    "saturation_mask",
    "summarize_mask",
]
