"""Array-level waveform feature extraction."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

Window = Optional[Union[slice, Tuple[int, int]]]


def _window_slice(n_samples: int, window: Window) -> slice:
    if window is None:
        return slice(None)
    if isinstance(window, tuple):
        if len(window) != 2:
            raise ValueError("window tuple must be (start, stop)")
        window = slice(window[0], window[1])
    if not isinstance(window, slice):
        raise TypeError("window must be None, slice, or (start, stop)")
    start, stop, step = window.indices(n_samples)
    if step != 1:
        raise ValueError("window step must be 1")
    if start >= stop:
        raise ValueError(f"empty window: start={start}, stop={stop}")
    return slice(start, stop)


def _channel_view(waveforms: np.ndarray, ch_idx: int, window: Window) -> np.ndarray:
    data = np.asarray(waveforms)
    if data.ndim != 3:
        raise ValueError(f"waveforms must be shaped (time, channel, event), got {data.shape}")
    n_samples, n_channels, _ = data.shape
    if ch_idx < 0 or ch_idx >= n_channels:
        raise IndexError(f"ch_idx={ch_idx} outside [0, {n_channels - 1}]")
    return data[_window_slice(n_samples, window), ch_idx, :]


def compute_max_in_window(waveforms: np.ndarray, ch_idx: int, window: Window = None) -> np.ndarray:
    return np.max(_channel_view(waveforms, ch_idx, window), axis=0).astype(np.float64)


def compute_min_in_window(waveforms: np.ndarray, ch_idx: int, window: Window = None) -> np.ndarray:
    return np.min(_channel_view(waveforms, ch_idx, window), axis=0).astype(np.float64)


def pedestal_mean_var(
    waveforms: np.ndarray,
    *,
    ch_idx: int,
    n_samples: int = 500,
    side: str = "front",
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-event pedestal mean and variance for a channel."""

    data = np.asarray(waveforms)
    if data.ndim != 3:
        raise ValueError(f"waveforms must be shaped (time, channel, event), got {data.shape}")
    n = min(int(n_samples), data.shape[0])
    if n <= 0:
        raise ValueError("n_samples must be positive")
    if side == "front":
        seg = data[:n, ch_idx, :]
    elif side == "back":
        seg = data[-n:, ch_idx, :]
    else:
        raise ValueError("side must be 'front' or 'back'")
    seg = np.asarray(seg, dtype=np.float64)
    return seg.mean(axis=0), seg.var(axis=0)


def time_axis(n_samples: int, *, sampling_interval_ns: float = 4.0, unit: str = "us") -> np.ndarray:
    scale = {"ns": 1.0, "us": 1e-3, "ms": 1e-6, "s": 1e-9}
    if unit not in scale:
        raise ValueError(f"Unsupported time unit {unit!r}")
    return np.arange(int(n_samples), dtype=np.float64) * float(sampling_interval_ns) * scale[unit]


__all__ = [
    "compute_max_in_window",
    "compute_min_in_window",
    "pedestal_mean_var",
    "time_axis",
]
