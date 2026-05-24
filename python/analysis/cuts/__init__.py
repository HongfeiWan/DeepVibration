"""Reusable physics cut definitions."""

from .acv import acv_mask, act_mask
from .basic import (
    CutResult,
    inhibit_mask,
    pedestal_3sigma_mask,
    rt_mask,
    saturation_mask,
    summarize_mask,
)
from .legacy import CH3PED_MIN_X_RANGE, bscut_mask, ch3ped_min_mask, fit_success_mask
from .mincut import mincut_mask
from .pn import pncut_mask
from .time import (
    CH0_BAND_BURST_HI,
    CH0_BAND_BURST_LO,
    CH0_TIME_BAND_BURST_RATE_THRESHOLD,
    CH0_TIME_EXCLUDE_BAD_INTERVALS_MPL,
    build_ch0_time_exclude_intervals_global,
    compute_ch0_time_exclude_intervals_mpl,
    cut_time,
    time_exclusion_mask,
)

__all__ = [
    "CH0_BAND_BURST_HI",
    "CH0_BAND_BURST_LO",
    "CH0_TIME_BAND_BURST_RATE_THRESHOLD",
    "CH0_TIME_EXCLUDE_BAD_INTERVALS_MPL",
    "CH3PED_MIN_X_RANGE",
    "CutResult",
    "acv_mask",
    "act_mask",
    "bscut_mask",
    "build_ch0_time_exclude_intervals_global",
    "compute_ch0_time_exclude_intervals_mpl",
    "cut_time",
    "ch3ped_min_mask",
    "fit_success_mask",
    "inhibit_mask",
    "mincut_mask",
    "pedestal_3sigma_mask",
    "pncut_mask",
    "rt_mask",
    "saturation_mask",
    "summarize_mask",
    "time_exclusion_mask",
]
