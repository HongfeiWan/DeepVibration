"""Anti-coincidence veto and ACT masks."""

from __future__ import annotations

import numpy as np


def acv_mask(
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    *,
    trigger_threshold: float = 7060.0,
    t_ge_us: float = 40.0,
    sampling_interval_ns: float = 4.0,
    dt_min_us: float = 1.0,
    dt_max_us: float = 16.0,
) -> np.ndarray:
    """Return True for ACV events: no NaI trigger, or NaI trigger outside the coincidence window."""

    max_arr = np.asarray(max_ch4, dtype=np.float64).reshape(-1)
    tmax_arr = np.asarray(tmax_ch4, dtype=np.float64).reshape(-1)
    n = min(max_arr.size, tmax_arr.size)
    max_arr = max_arr[:n]
    tmax_arr = tmax_arr[:n]
    nai_triggered = max_arr >= float(trigger_threshold)
    t_ch4_us = tmax_arr * float(sampling_interval_ns) * 1e-3
    delta_t_us = float(t_ge_us) - t_ch4_us
    outside_coincidence = (delta_t_us < float(dt_min_us)) | (delta_t_us > float(dt_max_us))
    return (~nai_triggered) | (nai_triggered & outside_coincidence)


def act_mask(*args, **kwargs) -> np.ndarray:
    """Return True for ACT events: NaI over threshold and ``delta_t`` inside ``[dt_min, dt_max]``."""

    return ~acv_mask(*args, **kwargs)


__all__ = ["acv_mask", "act_mask"]
