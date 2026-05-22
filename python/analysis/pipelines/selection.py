"""Reusable event-selection pipelines."""

from __future__ import annotations

from typing import Optional

import numpy as np

from analysis.cuts import CutResult, acv_mask, act_mask
from analysis.cuts.basic import combine_steps, physical_mask, saturation_mask


def run_basic_selection(
    *,
    max_ch5: np.ndarray,
    ch0_min: np.ndarray,
    max_ch0: Optional[np.ndarray] = None,
    max_ch1: Optional[np.ndarray] = None,
    rt_threshold: float = 6000.0,
    max_adc: float = 16382.0,
) -> CutResult:
    """Run the common RT/Inhibit/saturation preselection."""

    steps = {
        "physical": physical_mask(max_ch5, ch0_min, rt_threshold=rt_threshold),
    }
    if max_ch0 is not None:
        steps["not_saturated"] = saturation_mask(max_ch0, max_ch1, max_adc=max_adc)
    return combine_steps(steps)


def run_physical_selection(**kwargs) -> CutResult:
    """Alias for the baseline physical event selection."""

    return run_basic_selection(**kwargs)


def run_act_acv_selection(
    *,
    max_ch5: np.ndarray,
    ch0_min: np.ndarray,
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    max_ch0: Optional[np.ndarray] = None,
    max_ch1: Optional[np.ndarray] = None,
    mode: str = "act",
    rt_threshold: float = 6000.0,
    max_adc: float = 16382.0,
) -> CutResult:
    """Run basic selection plus ACT or ACV classification."""

    base = run_basic_selection(
        max_ch5=max_ch5,
        ch0_min=ch0_min,
        max_ch0=max_ch0,
        max_ch1=max_ch1,
        rt_threshold=rt_threshold,
        max_adc=max_adc,
    )
    mode_l = mode.lower()
    if mode_l == "act":
        extra = act_mask(max_ch4, tmax_ch4)
    elif mode_l == "acv":
        extra = acv_mask(max_ch4, tmax_ch4)
    else:
        raise ValueError("mode must be 'act' or 'acv'")
    steps = dict(base.steps)
    steps[mode_l] = extra
    return combine_steps(steps)


__all__ = ["run_act_acv_selection", "run_basic_selection", "run_physical_selection"]
