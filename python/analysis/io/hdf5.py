"""Small HDF5 utilities for event-wise analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from analysis.parallel import chunk_ranges

StatName = Literal["max", "min", "mean"]


def read_dataset(path: str | Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise KeyError(f"{path} does not contain dataset {key!r}")
        return np.asarray(handle[key][...])


def read_waveform(
    path: str | Path,
    *,
    event_idx: int,
    channel_idx: int = 0,
    dataset: str = "channel_data",
) -> np.ndarray:
    """Read one waveform from a ``(time, channel, event)`` dataset."""

    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"{path} does not contain dataset {dataset!r}")
        data = handle[dataset]
        if data.ndim != 3:
            raise ValueError(f"{dataset} must be 3D (time, channel, event), got {data.shape}")
        n_time, n_channels, n_events = data.shape
        if channel_idx < 0 or channel_idx >= n_channels:
            raise IndexError(f"channel_idx={channel_idx} outside [0, {n_channels - 1}]")
        if event_idx < 0 or event_idx >= n_events:
            raise IndexError(f"event_idx={event_idx} outside [0, {n_events - 1}]")
        return np.asarray(data[:, channel_idx, event_idx])


def compute_channel_stat(
    path: str | Path,
    *,
    channel_idx: int = 0,
    stat: StatName = "max",
    dataset: str = "channel_data",
    chunk_size: int = 1000,
) -> np.ndarray:
    """Compute a per-event statistic from a channel without loading all events."""

    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"{path} does not contain dataset {dataset!r}")
        data = handle[dataset]
        if data.ndim != 3:
            raise ValueError(f"{dataset} must be 3D (time, channel, event), got {data.shape}")
        _, n_channels, n_events = data.shape
        if channel_idx < 0 or channel_idx >= n_channels:
            raise IndexError(f"channel_idx={channel_idx} outside [0, {n_channels - 1}]")
        out = np.empty(n_events, dtype=np.float64)
        for start, stop in chunk_ranges(n_events, chunk_size):
            block = data[:, channel_idx, start:stop]
            if stat == "max":
                out[start:stop] = np.max(block, axis=0)
            elif stat == "min":
                out[start:stop] = np.min(block, axis=0)
            elif stat == "mean":
                out[start:stop] = np.mean(block, axis=0)
            else:
                raise ValueError(f"Unsupported stat: {stat}")
        return out


__all__ = ["compute_channel_stat", "read_dataset", "read_waveform"]
