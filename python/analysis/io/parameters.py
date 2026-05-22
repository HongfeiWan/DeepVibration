"""Parameter-file loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping

import h5py
import numpy as np


def load_parameter_file(path: str | Path, keys: Iterable[str]) -> Dict[str, np.ndarray]:
    """Load selected one-dimensional arrays from a parameter HDF5 file."""

    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        for key in keys:
            if key not in handle:
                raise KeyError(f"{path} does not contain dataset {key!r}")
            out[key] = np.asarray(handle[key][...])
    return out


def load_parameter_stack(files_to_keys: Mapping[str | Path, Iterable[str]]) -> Dict[str, np.ndarray]:
    """Load and merge datasets from several parameter files into one dict."""

    merged: Dict[str, np.ndarray] = {}
    for path, keys in files_to_keys.items():
        for key, value in load_parameter_file(path, keys).items():
            if key in merged:
                raise KeyError(f"Duplicate parameter key requested: {key}")
            merged[key] = value
    return merged


__all__ = ["load_parameter_file", "load_parameter_stack"]
