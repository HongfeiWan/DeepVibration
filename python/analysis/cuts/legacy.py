"""Legacy physics cut helpers kept for backward-compatible cut flows."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

CH3PED_MIN_X_RANGE: Tuple[float, float] = (960.0, 980.0)
CH3PED_X_MEAN_BAND_HALF_SIGMA: float = 0.5
CH3PED_RESIDUAL_N_SIGMA: float = 6.0
_LN_19 = float(np.log(19.0))


def _common_length(*arrays: np.ndarray) -> int:
    if not arrays:
        return 0
    return min(np.asarray(arr).size for arr in arrays)


def fit_success_mask(
    ch2_n_fit_points: np.ndarray,
    ch3_n_fit_points: np.ndarray,
    ch2_tanh_p0: np.ndarray,
    ch3_tanh_p0: np.ndarray,
    *,
    bad_val: float = 1e6,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, float]]:
    """Keep events whose CH2/CH3 fits converged and did not return sentinel values."""

    ch2_n = np.asarray(ch2_n_fit_points, dtype=np.int32).reshape(-1)
    ch3_n = np.asarray(ch3_n_fit_points, dtype=np.int32).reshape(-1)
    ch2_p0 = np.asarray(ch2_tanh_p0, dtype=np.float64).reshape(-1)
    ch3_p0 = np.asarray(ch3_tanh_p0, dtype=np.float64).reshape(-1)
    n = _common_length(ch2_n, ch3_n, ch2_p0, ch3_p0)
    ch2_n = ch2_n[:n]
    ch3_n = ch3_n[:n]
    ch2_p0 = ch2_p0[:n]
    ch3_p0 = ch3_p0[:n]

    ok_npts = (ch2_n > 0) & (ch3_n > 0)
    ok_ch2 = np.isfinite(ch2_p0) & (~np.isclose(ch2_p0, float(bad_val)))
    ok_ch3 = np.isfinite(ch3_p0) & (~np.isclose(ch3_p0, float(bad_val)))
    keep = ok_npts & ok_ch2 & ok_ch3

    if not return_stats:
        return keep

    stats: Dict[str, float] = {
        "total": float(n),
        "fit_events": float(np.count_nonzero(ok_npts)),
        "ch2_ok": float(np.count_nonzero(ok_ch2)),
        "ch3_ok": float(np.count_nonzero(ok_ch3)),
        "passed": float(np.count_nonzero(keep)),
    }
    return keep, stats


def ch3ped_min_mask(
    ch3ped_mean: np.ndarray,
    min_ch3: np.ndarray,
    *,
    sigma_yx: float = 20.0,
    x_range: Tuple[float, float] = CH3PED_MIN_X_RANGE,
    x_mean_band_half_sigma: float = CH3PED_X_MEAN_BAND_HALF_SIGMA,
    n_sigma_residual: float = CH3PED_RESIDUAL_N_SIGMA,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, float]]:
    """Keep CH3 ped/min events inside the legacy red/green bands."""

    x = np.asarray(ch3ped_mean, dtype=np.float64).reshape(-1)
    y = np.asarray(min_ch3, dtype=np.float64).reshape(-1)
    n = _common_length(x, y)
    x = x[:n]
    y = y[:n]
    finite = np.isfinite(x) & np.isfinite(y)
    x_lo, x_hi = map(float, x_range)
    x_all = x[finite]
    if x_all.size >= 2:
        x_mean_all = float(np.mean(x_all))
        sigma_x = float(np.std(x_all, ddof=1))
        if np.isfinite(sigma_x) and sigma_x > 0.0:
            half_width = float(x_mean_band_half_sigma) * sigma_x
            red_ok = (x >= x_mean_all - half_width) & (x <= x_mean_all + half_width)
        else:
            red_ok = np.ones(n, dtype=bool)
    else:
        red_ok = np.ones(n, dtype=bool)

    in_yx_band = finite & (np.abs(y - x) <= float(sigma_yx))
    xf = x[in_yx_band]
    yf = y[in_yx_band]
    if xf.size >= 2:
        b_fit = float(np.mean(yf - xf))
        resid_fit = yf - xf - b_fit
        sigma_res = float(np.std(resid_fit, ddof=1))
        if np.isfinite(sigma_res) and sigma_res > 0.0:
            band_ok = np.abs(y - x - b_fit) <= float(n_sigma_residual) * sigma_res
        else:
            band_ok = np.ones(n, dtype=bool)
    else:
        band_ok = np.ones(n, dtype=bool)

    keep = finite & (x >= x_lo) & (x <= x_hi) & (y > 0.0) & red_ok & band_ok

    if not return_stats:
        return keep

    stats: Dict[str, float] = {
        "total": float(n),
        "finite": float(np.count_nonzero(finite)),
        "passed": float(np.count_nonzero(keep)),
        "x_lo": x_lo,
        "x_hi": x_hi,
        "x_mean_band_half_sigma": float(x_mean_band_half_sigma),
        "n_sigma_residual": float(n_sigma_residual),
    }
    return keep, stats


def bscut_mask(
    tanh_p1: np.ndarray,
    *,
    rise_time_max_us: float = 0.8,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, float]]:
    """Keep CH3 events whose rise time ``ln(19) / p1`` stays below the legacy limit."""

    p1 = np.asarray(tanh_p1, dtype=np.float64).reshape(-1)
    p1_safe = np.where(p1 > 1e-10, p1, np.nan)
    rise_us = np.where(np.isfinite(p1_safe), _LN_19 / p1_safe, np.nan)
    keep = np.isfinite(rise_us) & (rise_us <= float(rise_time_max_us))

    if not return_stats:
        return keep

    stats: Dict[str, float] = {
        "total": float(p1.size),
        "passed": float(np.count_nonzero(keep)),
        "rise_time_max_us": float(rise_time_max_us),
    }
    return keep, stats


__all__ = [
    "CH3PED_MIN_X_RANGE",
    "CH3PED_RESIDUAL_N_SIGMA",
    "CH3PED_X_MEAN_BAND_HALF_SIGMA",
    "bscut_mask",
    "ch3ped_min_mask",
    "fit_success_mask",
]
