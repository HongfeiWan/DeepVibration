"""Event-level feature cache, Mahalanobis metric, and UMAP diagnostics."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from analysis.cuts import (
    acv_mask,
    act_mask,
    fit_success_mask,
    inhibit_mask,
    pedestal_3sigma_mask,
    pncut_mask,
    rt_mask,
    saturation_mask,
    time_exclusion_mask,
)
from analysis.io.paths import RunFiles, find_project_root, pair_parameter_files, raw_pulse_dir
from analysis.parallel import ParallelConfig, iter_completed


NUMERIC_KINDS = {"b", "i", "u", "f"}
DEFAULT_CACHE_NAME = "event_feature_cache.h5"


@dataclass(frozen=True)
class FeatureSchema:
    channels: tuple[str, ...]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class RunFeatureMeta:
    name: str
    n_events: int
    feature_names: tuple[str, ...]


def _channel_sort_key(label: str) -> tuple[int, str]:
    text = label.upper().replace("_PARAMETERS", "")
    if text.startswith("CH") and text[2:].isdigit():
        return int(text[2:]), text
    return 999, text


def discover_parameter_channels(project_root: str | Path | None = None) -> list[str]:
    base = raw_pulse_dir(project_root)
    labels: list[str] = []
    if not base.exists():
        return labels
    for path in base.iterdir():
        if path.is_dir() and path.name.upper().endswith("_PARAMETERS"):
            labels.append(path.name[: -len("_parameters")].upper())
    return sorted(labels, key=_channel_sort_key)


def paired_parameter_runs(
    *,
    project_root: str | Path | None = None,
    channels: Sequence[str] | None = None,
    max_runs: int | None = None,
) -> list[RunFiles]:
    labels = list(channels) if channels is not None else discover_parameter_channels(project_root)
    if not labels:
        raise FileNotFoundError("No CH*_parameters folders found under data/hdf5/raw_pulse")
    runs = pair_parameter_files(labels, project_root=project_root)
    if max_runs is not None:
        runs = runs[: int(max_runs)]
    if not runs:
        raise FileNotFoundError(f"No paired parameter files found for channels: {labels}")
    return runs


def _inspect_numeric_1d_datasets(path: Path, prefix: str) -> tuple[list[str], int]:
    names: list[str] = []
    n_events = 0
    with h5py.File(path, "r") as handle:
        for key in sorted(handle.keys()):
            dset = handle[key]
            if not isinstance(dset, h5py.Dataset):
                continue
            if dset.ndim != 1 or dset.dtype.kind not in NUMERIC_KINDS or dset.size == 0:
                continue
            if n_events == 0:
                n_events = int(dset.shape[0])
            elif int(dset.shape[0]) != n_events:
                continue
            names.append(f"{prefix}{key}")
    return names, n_events


def inspect_feature_schema(runs: Sequence[RunFiles]) -> tuple[FeatureSchema, list[RunFeatureMeta]]:
    feature_union: set[str] = set()
    run_meta: list[RunFeatureMeta] = []
    channels = tuple(sorted(next(iter(runs)).files.keys(), key=_channel_sort_key))
    for run in runs:
        names: list[str] = []
        n_events: int | None = None
        for channel in channels:
            prefix = f"{channel.lower()}_"
            channel_names, channel_events = _inspect_numeric_1d_datasets(run.files[channel], prefix)
            if channel_events <= 0:
                continue
            if n_events is None:
                n_events = channel_events
            elif channel_events != n_events:
                raise ValueError(f"{run.name}: event count mismatch in {channel}")
            names.extend(channel_names)
        if n_events is None:
            raise ValueError(f"{run.name}: no numeric 1D parameter datasets found")
        feature_union.update(names)
        run_meta.append(RunFeatureMeta(run.name, n_events, tuple(sorted(names))))
    schema = FeatureSchema(channels=channels, feature_names=tuple(sorted(feature_union)))
    return schema, run_meta


def _read_run_feature_block(
    run_name: str,
    files: Mapping[str, str],
    feature_names: Sequence[str],
    *,
    project_root: str,
    time_epoch_offset: float,
    time_unit: str,
) -> dict[str, np.ndarray | str | int]:
    feature_index = {name: i for i, name in enumerate(feature_names)}
    n_events: int | None = None
    matrix: np.ndarray | None = None

    for channel in sorted(files, key=_channel_sort_key):
        path = Path(files[channel])
        prefix = f"{channel.lower()}_"
        with h5py.File(path, "r") as handle:
            for key in sorted(handle.keys()):
                dset = handle[key]
                if not isinstance(dset, h5py.Dataset):
                    continue
                if dset.ndim != 1 or dset.dtype.kind not in NUMERIC_KINDS or dset.size == 0:
                    continue
                if n_events is None:
                    n_events = int(dset.shape[0])
                    matrix = np.full((n_events, len(feature_names)), np.nan, dtype=np.float32)
                elif int(dset.shape[0]) != n_events:
                    continue
                name = f"{prefix}{key}"
                idx = feature_index.get(name)
                if idx is None:
                    continue
                data = np.asarray(dset[...], dtype=np.float32)
                assert matrix is not None
                matrix[:, idx] = data

    if n_events is None or matrix is None:
        raise ValueError(f"{run_name}: no parameter arrays were loaded")

    raw_time = np.full(n_events, np.nan, dtype=np.float64)
    ch03 = Path(project_root) / "data" / "hdf5" / "raw_pulse" / "CH0-3" / run_name
    if ch03.exists():
        with h5py.File(ch03, "r") as handle:
            if "time_data" in handle:
                data = np.asarray(handle["time_data"][:n_events], dtype=np.float64)
                if time_unit == "unix":
                    data = data - float(time_epoch_offset)
                raw_time[: data.size] = data

    return {
        "run_name": run_name,
        "n_events": n_events,
        "features": matrix,
        "time": raw_time,
    }


def _read_run_feature_block_item(item):
    return _read_run_feature_block(**item)


def _parse_ranges(ranges: Sequence[tuple[float, float]] | None) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, stop in ranges or []:
        a = float(start)
        b = float(stop)
        if np.isfinite(a) and np.isfinite(b) and b > a:
            out.append((a, b))
    return out


def _columns_from_matrix(matrix: np.ndarray, feature_names: Sequence[str]) -> dict[str, np.ndarray]:
    index = {name: i for i, name in enumerate(feature_names)}
    return {name: matrix[:, i] for name, i in index.items()}


def compute_event_masks(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    event_time: np.ndarray,
    *,
    pn_fit_ch0_min: float = 5000.0,
    pn_fit_ch0_max: float = 16000.0,
    pn_sigma: float = 0.8,
    time_keep_ranges: Sequence[tuple[float, float]] | None = None,
    bad_time_intervals: Sequence[tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    cols = _columns_from_matrix(matrix, feature_names)
    n = int(matrix.shape[0])

    def col(name: str, default: float = np.nan) -> np.ndarray:
        return np.asarray(cols.get(name, np.full(n, default, dtype=np.float32)), dtype=np.float64)

    max_ch5 = col("ch5_max_ch5")
    ch0_min = col("ch0_ch0_min")
    max_ch0 = col("ch0_max_ch0")
    max_ch1 = col("ch1_max_ch1")
    max_ch4 = col("ch4_max_ch4")
    tmax_ch4 = col("ch4_tmax_ch4")
    ch0_ped = col("ch0_ch0ped_mean")
    ch1_ped = col("ch1_ch1ped_mean")

    rt = rt_mask(max_ch5)
    inhibit = inhibit_mask(ch0_min)
    physical = (~rt) & (~inhibit)
    pedestal_ok = np.asarray(pedestal_3sigma_mask(ch0_ped, ch1_ped, rt), dtype=bool)
    pedestal_outlier = physical & (~pedestal_ok)
    saturated_ok = saturation_mask(max_ch0, max_ch1)
    over_threshold = physical & pedestal_ok & (~saturated_ok)
    acv = acv_mask(max_ch4, tmax_ch4)
    act = act_mask(max_ch4, tmax_ch4)
    base = physical & pedestal_ok & saturated_ok
    pn = np.asarray(
        pncut_mask(
            max_ch0,
            max_ch1,
            base_mask=base,
            fit_ch0_min=pn_fit_ch0_min,
            fit_ch0_max=pn_fit_ch0_max,
            n_sigma=pn_sigma,
        ),
        dtype=bool,
    )

    time_keep = np.ones(n, dtype=bool)
    finite_time = np.isfinite(event_time)
    keep_ranges = _parse_ranges(time_keep_ranges)
    if keep_ranges:
        time_keep = np.zeros(n, dtype=bool)
        for start, stop in keep_ranges:
            time_keep |= (event_time >= start) & (event_time < stop)
        time_keep &= finite_time
    bad_ranges = _parse_ranges(bad_time_intervals)
    if bad_ranges:
        time_keep &= time_exclusion_mask(event_time, bad_ranges)

    not_pn = base & (~pn)
    not_time = base & (~time_keep)
    clean = base & acv & pn & time_keep

    labels = np.zeros(n, dtype=np.int16)
    label_names = [
        "clean_or_other",
        "rt",
        "inhibit",
        "pedestal_outlier",
        "over_threshold",
        "act",
        "not_pn",
        "not_time",
    ]
    for code, mask in (
        (1, rt),
        (2, inhibit),
        (3, pedestal_outlier),
        (4, over_threshold),
        (5, base & act),
        (6, not_pn),
        (7, not_time),
    ):
        labels[(labels == 0) & mask] = code

    return {
        "rt": rt,
        "inhibit": inhibit,
        "physical": physical,
        "pedestal": pedestal_ok,
        "pedestal_outlier": pedestal_outlier,
        "saturation": saturated_ok,
        "over_threshold": over_threshold,
        "basic": base,
        "acv": acv,
        "act": base & act,
        "pn": pn,
        "not_pn": not_pn,
        "time_keep": time_keep,
        "not_time": not_time,
        "clean": clean,
        "event_class": labels,
        "event_class_names": np.asarray(label_names, dtype=h5py.string_dtype("utf-8")),
    }


def build_feature_cache(
    *,
    project_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    channels: Sequence[str] | None = None,
    workers: int | str | None = "auto",
    chunk_size: int = 10000,
    compression: str | None = "lzf",
    rebuild: bool = False,
    max_runs: int | None = None,
    pn_fit_ch0_min: float = 5000.0,
    pn_fit_ch0_max: float = 16000.0,
    pn_sigma: float = 0.8,
    time_keep_ranges: Sequence[tuple[float, float]] | None = None,
    bad_time_intervals: Sequence[tuple[float, float]] | None = None,
    time_epoch_offset: float = 2.082816000000000e09,
    time_unit: str = "raw",
) -> Path:
    root = find_project_root(project_root)
    out_dir = Path(cache_dir) if cache_dir is not None else root / "data" / "cache" / "event_feature_umap"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / DEFAULT_CACHE_NAME
    if cache_path.exists() and not rebuild:
        return cache_path
    if cache_path.exists():
        cache_path.unlink()

    runs = paired_parameter_runs(project_root=root, channels=channels, max_runs=max_runs)
    schema, run_meta = inspect_feature_schema(runs)
    total_events = int(sum(meta.n_events for meta in run_meta))
    feature_names = list(schema.feature_names)
    feature_count = len(feature_names)
    compression_arg = None if compression in {None, "none", "None"} else compression

    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["project_root"] = str(root)
        handle.attrs["schema_version"] = "event_feature_cache_v1"
        handle.attrs["channels"] = json.dumps(schema.channels)
        handle.attrs["total_events"] = total_events
        handle.attrs["n_features"] = feature_count
        handle.create_dataset("feature_names", data=np.asarray(feature_names, dtype=string_dtype))
        handle.create_dataset("run_names", data=np.asarray([m.name for m in run_meta], dtype=string_dtype))
        handle.create_dataset("run_event_counts", data=np.asarray([m.n_events for m in run_meta], dtype=np.int64))
        handle.create_dataset("features", shape=(total_events, feature_count), dtype=np.float32,
                              chunks=(min(chunk_size, max(1, total_events)), feature_count),
                              compression=compression_arg)
        handle.create_dataset("run_index", shape=(total_events,), dtype=np.int32,
                              chunks=(min(chunk_size, max(1, total_events)),), compression=compression_arg)
        handle.create_dataset("event_index", shape=(total_events,), dtype=np.int32,
                              chunks=(min(chunk_size, max(1, total_events)),), compression=compression_arg)
        handle.create_dataset("event_time", shape=(total_events,), dtype=np.float64,
                              chunks=(min(chunk_size, max(1, total_events)),), compression=compression_arg)
        mask_group = handle.create_group("masks")
        for name in (
            "rt",
            "inhibit",
            "physical",
            "pedestal",
            "pedestal_outlier",
            "saturation",
            "over_threshold",
            "basic",
            "acv",
            "act",
            "pn",
            "not_pn",
            "time_keep",
            "not_time",
            "clean",
        ):
            mask_group.create_dataset(name, shape=(total_events,), dtype=np.uint8,
                                      chunks=(min(chunk_size, max(1, total_events)),), compression=compression_arg)
        mask_group.create_dataset("event_class", shape=(total_events,), dtype=np.int16,
                                  chunks=(min(chunk_size, max(1, total_events)),), compression=compression_arg)
        mask_group.create_dataset("event_class_names", data=np.asarray(
            [
                "clean_or_other",
                "rt",
                "inhibit",
                "pedestal_outlier",
                "over_threshold",
                "act",
                "not_pn",
                "not_time",
            ],
            dtype=string_dtype,
        ))

        offsets: dict[str, tuple[int, int, int]] = {}
        start = 0
        for run_idx, meta in enumerate(run_meta):
            stop = start + meta.n_events
            offsets[meta.name] = (run_idx, start, stop)
            start = stop

        tasks = [
            {
                "run_name": run.name,
                "files": {channel: str(path) for channel, path in run.files.items()},
                "feature_names": feature_names,
                "project_root": str(root),
                "time_epoch_offset": time_epoch_offset,
                "time_unit": time_unit,
            }
            for run in runs
        ]
        cfg = ParallelConfig(workers=workers, chunk_size=chunk_size)
        for _, block in iter_completed(_read_run_feature_block_item, tasks, config=cfg):
            run_name = str(block["run_name"])
            run_idx, start, stop = offsets[run_name]
            matrix = np.asarray(block["features"], dtype=np.float32)
            event_time = np.asarray(block["time"], dtype=np.float64)
            if matrix.shape[0] != stop - start:
                raise ValueError(f"{run_name}: feature rows changed during cache build")
            handle["features"][start:stop, :] = matrix
            handle["run_index"][start:stop] = run_idx
            handle["event_index"][start:stop] = np.arange(matrix.shape[0], dtype=np.int32)
            handle["event_time"][start:stop] = event_time
            masks = compute_event_masks(
                matrix,
                feature_names,
                event_time,
                pn_fit_ch0_min=pn_fit_ch0_min,
                pn_fit_ch0_max=pn_fit_ch0_max,
                pn_sigma=pn_sigma,
                time_keep_ranges=time_keep_ranges,
                bad_time_intervals=bad_time_intervals,
            )
            for name, value in masks.items():
                if name == "event_class_names":
                    continue
                data = np.asarray(value)
                if data.dtype == bool:
                    data = data.astype(np.uint8)
                mask_group[name][start:stop] = data

    return cache_path


def load_feature_names(cache_path: str | Path) -> list[str]:
    with h5py.File(cache_path, "r") as handle:
        return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in handle["feature_names"][...]]


def _chunk_ranges(n_items: int, chunk_size: int):
    for start in range(0, n_items, chunk_size):
        yield start, min(start + chunk_size, n_items)


def compute_feature_mean_scale(cache_path: str | Path, *, chunk_size: int = 100000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(cache_path, "a") as handle:
        features = handle["features"]
        n_events, n_features = features.shape
        total = np.zeros(n_features, dtype=np.float64)
        count = np.zeros(n_features, dtype=np.float64)
        for start, stop in _chunk_ranges(n_events, chunk_size):
            block = np.asarray(features[start:stop], dtype=np.float64)
            finite = np.isfinite(block)
            total += np.where(finite, block, 0.0).sum(axis=0)
            count += finite.sum(axis=0)
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        ssq = np.zeros(n_features, dtype=np.float64)
        for start, stop in _chunk_ranges(n_events, chunk_size):
            block = np.asarray(features[start:stop], dtype=np.float64)
            finite = np.isfinite(block)
            diff = np.where(finite, block - mean, 0.0)
            ssq += (diff * diff).sum(axis=0)
        var = np.divide(ssq, np.maximum(count - 1.0, 1.0), out=np.ones_like(ssq), where=count > 1)
        scale = np.sqrt(np.maximum(var, 1e-12))
        group = handle.require_group("preprocessing")
        for name, data in (("feature_mean", mean), ("feature_scale", scale), ("feature_finite_count", count)):
            if name in group:
                del group[name]
            group.create_dataset(name, data=data)
        return mean, scale, count


def _standardize_block(block: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = np.asarray(block, dtype=np.float64)
    x = np.where(np.isfinite(x), x, mean)
    x = (x - mean) / scale
    return x


def _sample_indices(n_events: int, max_events: int, seed: int) -> np.ndarray:
    if max_events <= 0 or n_events <= max_events:
        return np.arange(n_events, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_events, size=max_events, replace=False).astype(np.int64))


def _sample_indices_with_priority(
    n_events: int,
    max_events: int,
    seed: int,
    priority_indices: np.ndarray | None = None,
) -> np.ndarray:
    if max_events <= 0 or n_events <= max_events:
        return np.arange(n_events, dtype=np.int64)
    if priority_indices is None:
        return _sample_indices(n_events, max_events, seed)

    priority = np.asarray(priority_indices, dtype=np.int64).reshape(-1)
    priority = priority[(priority >= 0) & (priority < n_events)]
    if priority.size == 0:
        return _sample_indices(n_events, max_events, seed)
    priority = np.unique(priority)

    rng = np.random.default_rng(seed)
    if priority.size >= max_events:
        return np.sort(rng.choice(priority, size=max_events, replace=False).astype(np.int64))

    selected = np.zeros(n_events, dtype=bool)
    selected[priority] = True
    remaining_pool = np.flatnonzero(~selected).astype(np.int64)
    need = int(max_events) - int(priority.size)
    fill = rng.choice(remaining_pool, size=need, replace=False).astype(np.int64)
    return np.sort(np.concatenate([priority, fill]))


def _priority_indices_from_masks(
    handle: h5py.File,
    group_name: str,
    mask_names: Sequence[str],
    *,
    chunk_size: int,
    max_events_per_mask: int | None,
    random_seed: int,
) -> np.ndarray:
    group = handle.get(group_name.strip("/"))
    if group is None:
        raise KeyError(f"Priority mask group does not exist: {group_name}")
    rng = np.random.default_rng(random_seed)
    pieces: list[np.ndarray] = []
    for mask_name in mask_names:
        if mask_name not in group:
            raise KeyError(f"Priority mask does not exist: {group_name}/{mask_name}")
        dset = group[mask_name]
        if not isinstance(dset, h5py.Dataset) or dset.ndim != 1:
            raise ValueError(f"Priority mask must be a 1D dataset: {group_name}/{mask_name}")
        mask_indices: list[np.ndarray] = []
        for start, stop in _chunk_ranges(int(dset.shape[0]), chunk_size):
            block = np.asarray(dset[start:stop], dtype=bool)
            if np.any(block):
                mask_indices.append(np.flatnonzero(block).astype(np.int64) + start)
        if not mask_indices:
            continue
        indices = np.concatenate(mask_indices)
        if max_events_per_mask is not None and indices.size > int(max_events_per_mask):
            indices = rng.choice(indices, size=int(max_events_per_mask), replace=False).astype(np.int64)
        pieces.append(indices)
    if not pieces:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(pieces))


def _read_rows(dataset, indices: np.ndarray, *, chunk_size: int = 100000) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return np.empty((0, dataset.shape[1]), dtype=dataset.dtype)
    order = np.argsort(idx)
    sorted_idx = idx[order]
    out = np.empty((idx.size, dataset.shape[1]), dtype=dataset.dtype)
    n_events = int(dataset.shape[0])
    for start, stop in _chunk_ranges(n_events, chunk_size):
        left = int(np.searchsorted(sorted_idx, start, side="left"))
        right = int(np.searchsorted(sorted_idx, stop, side="left"))
        if right <= left:
            continue
        block = np.asarray(dataset[start:stop])
        out[order[left:right]] = block[sorted_idx[left:right] - start]
    return out


def _read_column_rows(dataset, indices: np.ndarray, columns: Sequence[int], *, chunk_size: int = 100000) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    cols = np.asarray(columns, dtype=np.int64)
    if idx.size == 0:
        return np.empty((0, cols.size), dtype=dataset.dtype)
    if cols.size == 0:
        raise ValueError("At least one feature column is required")
    order = np.argsort(idx)
    sorted_idx = idx[order]
    out = np.empty((idx.size, cols.size), dtype=dataset.dtype)
    n_events = int(dataset.shape[0])
    for start, stop in _chunk_ranges(n_events, chunk_size):
        left = int(np.searchsorted(sorted_idx, start, side="left"))
        right = int(np.searchsorted(sorted_idx, stop, side="left"))
        if right <= left:
            continue
        block = np.asarray(dataset[start:stop])
        out[order[left:right]] = block[sorted_idx[left:right] - start][:, cols]
    return out


def _read_vector(dataset, indices: np.ndarray, *, chunk_size: int = 100000) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return np.empty((0,), dtype=dataset.dtype)
    order = np.argsort(idx)
    sorted_idx = idx[order]
    out = np.empty((idx.size,), dtype=dataset.dtype)
    n_events = int(dataset.shape[0])
    for start, stop in _chunk_ranges(n_events, chunk_size):
        left = int(np.searchsorted(sorted_idx, start, side="left"))
        right = int(np.searchsorted(sorted_idx, stop, side="left"))
        if right <= left:
            continue
        block = np.asarray(dataset[start:stop])
        out[order[left:right]] = block[sorted_idx[left:right] - start]
    return out


def _thread_limit_context(workers: int | str | None):
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return nullcontext()
    cfg = ParallelConfig(workers=workers)
    return threadpool_limits(limits=cfg.resolved_workers())


def _load_or_compute_feature_mean_scale(
    cache_path: str | Path,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(cache_path, "r") as handle:
        group = handle.get("preprocessing")
        n_features = int(handle["features"].shape[1])
        if group is not None and {"feature_mean", "feature_scale", "feature_finite_count"} <= set(group.keys()):
            mean = np.asarray(group["feature_mean"][...], dtype=np.float64)
            scale = np.asarray(group["feature_scale"][...], dtype=np.float64)
            count = np.asarray(group["feature_finite_count"][...], dtype=np.float64)
            if mean.size == n_features and scale.size == n_features and count.size == n_features:
                return mean, scale, count
    return compute_feature_mean_scale(cache_path, chunk_size=chunk_size)


def _feature_columns_for_prefixes(feature_names: Sequence[str], prefixes: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    normalized = tuple(prefix.lower() for prefix in prefixes)
    columns = [
        idx
        for idx, name in enumerate(feature_names)
        if any(name.lower().startswith(prefix) for prefix in normalized)
    ]
    if not columns:
        raise ValueError(f"No feature names match prefixes: {list(prefixes)}")
    names = [feature_names[idx] for idx in columns]
    return np.asarray(columns, dtype=np.int64), names


def _read_feature_column(
    handle: h5py.File,
    feature_names: Sequence[str],
    name: str,
    *,
    chunk_size: int = 100000,
    default: float = np.nan,
) -> np.ndarray:
    features = handle["features"]
    n_events = int(features.shape[0])
    try:
        col = feature_names.index(name)
    except ValueError:
        return np.full(n_events, default, dtype=np.float64)
    out = np.empty(n_events, dtype=np.float64)
    for start, stop in _chunk_ranges(n_events, chunk_size):
        out[start:stop] = np.asarray(features[start:stop, col], dtype=np.float64)
    return out


def _sigma_outlier_mask(
    values: np.ndarray,
    fit_mask: np.ndarray,
    *,
    n_sigma: float,
    min_fit_events: int,
) -> tuple[np.ndarray, dict[str, float]]:
    arr = np.asarray(values, dtype=np.float64)
    fit = np.asarray(fit_mask, dtype=bool)
    finite = np.isfinite(arr) & fit
    stats: dict[str, float] = {"fit_events": float(np.count_nonzero(finite))}
    out = np.zeros(arr.size, dtype=bool)
    if np.count_nonzero(finite) < int(min_fit_events):
        stats.update({"used": 0.0, "mu": np.nan, "sigma": np.nan})
        return out, stats
    sample = arr[finite]
    mu = float(sample.mean())
    sigma = float(sample.std(ddof=1))
    stats.update({"used": 1.0, "mu": mu, "sigma": sigma})
    if sigma <= 0.0:
        return out, stats
    out = np.abs(arr - mu) > float(n_sigma) * sigma
    out &= np.isfinite(arr)
    return out, stats


def compute_basic_anomaly_masks(
    cache_path: str | Path,
    *,
    group_name: str = "anomaly_masks",
    rt_threshold: float = 6000.0,
    max_adc: float = 16382.0,
    n_sigma: float = 3.0,
    min_fit_events: int = 10,
    chunk_size: int = 100000,
) -> dict[str, float]:
    """Compute physics-motivated anomaly classes for UMAP demonstration plots."""

    with h5py.File(cache_path, "a") as handle:
        feature_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in handle["feature_names"][...]
        ]

        max_ch5 = _read_feature_column(handle, feature_names, "ch5_max_ch5", chunk_size=chunk_size)
        ch0_min = _read_feature_column(handle, feature_names, "ch0_ch0_min", chunk_size=chunk_size)
        ch1_min = _read_feature_column(handle, feature_names, "ch1_ch1_min", chunk_size=chunk_size)
        max_ch0 = _read_feature_column(handle, feature_names, "ch0_max_ch0", chunk_size=chunk_size)
        max_ch1 = _read_feature_column(handle, feature_names, "ch1_max_ch1", chunk_size=chunk_size)
        max_ch4 = _read_feature_column(handle, feature_names, "ch4_max_ch4", chunk_size=chunk_size)
        tmax_ch4 = _read_feature_column(handle, feature_names, "ch4_tmax_ch4", chunk_size=chunk_size)
        ch0_ped = _read_feature_column(handle, feature_names, "ch0_ch0ped_mean", chunk_size=chunk_size)
        ch1_ped = _read_feature_column(handle, feature_names, "ch1_ch1ped_mean", chunk_size=chunk_size)
        ch2_n_fit_points = _read_feature_column(
            handle,
            feature_names,
            "ch2_n_fit_points",
            chunk_size=chunk_size,
            default=0.0,
        )
        ch3_n_fit_points = _read_feature_column(
            handle,
            feature_names,
            "ch3_n_fit_points",
            chunk_size=chunk_size,
            default=0.0,
        )
        ch2_tanh_p0 = _read_feature_column(
            handle,
            feature_names,
            "ch2_tanh_p0",
            chunk_size=chunk_size,
            default=1e6,
        )
        ch3_tanh_p0 = _read_feature_column(
            handle,
            feature_names,
            "ch3_tanh_p0",
            chunk_size=chunk_size,
            default=1e6,
        )

        rt = rt_mask(max_ch5, threshold=rt_threshold)
        inhibit = inhibit_mask(ch0_min)
        fit_success, fit_stats = fit_success_mask(
            ch2_n_fit_points,
            ch3_n_fit_points,
            ch2_tanh_p0,
            ch3_tanh_p0,
            return_stats=True,
        )
        fit_success = np.asarray(fit_success, dtype=bool)
        fit_failed = ~fit_success
        physical = (~rt) & (~inhibit)

        pedestal_ok, pedestal_stats = pedestal_3sigma_mask(
            ch0_ped,
            ch1_ped,
            rt,
            n_sigma=n_sigma,
            min_fit_events=min_fit_events,
            return_stats=True,
        )
        pedestal_ok = np.asarray(pedestal_ok, dtype=bool)
        pedestal_3sigma = physical & (~pedestal_ok)
        after_pedestal = physical & pedestal_ok

        overthreshold = after_pedestal & ((max_ch0 > float(max_adc)) | (max_ch1 > float(max_adc)))
        saturated_basic = after_pedestal & (~overthreshold)

        acv = saturated_basic & acv_mask(max_ch4, tmax_ch4)
        act = saturated_basic & act_mask(max_ch4, tmax_ch4)
        min_fit = acv
        ch0_min_out, ch0_min_stats = _sigma_outlier_mask(
            ch0_min,
            min_fit,
            n_sigma=n_sigma,
            min_fit_events=min_fit_events,
        )
        ch1_min_out, ch1_min_stats = _sigma_outlier_mask(
            ch1_min,
            min_fit,
            n_sigma=n_sigma,
            min_fit_events=min_fit_events,
        )
        min_3sigma = saturated_basic & (ch0_min_out | ch1_min_out)

        labels = np.zeros(max_ch5.size, dtype=np.int16)
        label_names = [
            "background",
            "fit_failed",
            "inhibit",
            "random_trigger",
            "pedestal_3sigma_outlier",
            "over_threshold",
            "min_3sigma_outlier",
        ]
        for code, mask in (
            (1, fit_failed),
            (2, inhibit),
            (3, rt),
            (4, pedestal_3sigma),
            (5, overthreshold),
            (6, min_3sigma),
        ):
            labels[(labels == 0) & mask] = code

        parent = handle
        parts = [part for part in group_name.strip("/").split("/") if part]
        if not parts:
            raise ValueError("group_name must not be empty")
        for part in parts[:-1]:
            parent = parent.require_group(part)
        if parts[-1] in parent:
            del parent[parts[-1]]
        group = parent.create_group(parts[-1])
        string_dtype = h5py.string_dtype("utf-8")
        for name, mask in (
            ("fit_success", fit_success),
            ("fit_failed", fit_failed),
            ("inhibit", inhibit),
            ("random_trigger", rt),
            ("pedestal_3sigma_outlier", pedestal_3sigma),
            ("over_threshold", overthreshold),
            ("acv", acv),
            ("act", act),
            ("min_3sigma_outlier", min_3sigma),
        ):
            group.create_dataset(name, data=np.asarray(mask, dtype=np.uint8), compression="lzf")
        group.create_dataset("event_class", data=labels, compression="lzf")
        group.create_dataset("event_class_names", data=np.asarray(label_names, dtype=string_dtype))
        group.attrs["rt_threshold"] = float(rt_threshold)
        group.attrs["max_adc"] = float(max_adc)
        group.attrs["n_sigma"] = float(n_sigma)
        group.attrs["min_fit_events"] = int(min_fit_events)
        for key, value in fit_stats.items():
            group.attrs[f"fit_{key}"] = float(value)
        for prefix, stats in (("ch0_min", ch0_min_stats), ("ch1_min", ch1_min_stats)):
            for key, value in stats.items():
                group.attrs[f"{prefix}_{key}"] = float(value)
        for key, value in pedestal_stats.items():
            group.attrs[f"pedestal_{key}"] = float(value)

        return {
            "total_events": float(max_ch5.size),
            "fit_failed": float(np.count_nonzero(fit_failed)),
            "fit_success": float(np.count_nonzero(fit_success)),
            "inhibit": float(np.count_nonzero(inhibit)),
            "random_trigger": float(np.count_nonzero(rt)),
            "pedestal_3sigma_outlier": float(np.count_nonzero(pedestal_3sigma)),
            "over_threshold": float(np.count_nonzero(overthreshold)),
            "acv": float(np.count_nonzero(acv)),
            "act": float(np.count_nonzero(act)),
            "min_3sigma_outlier": float(np.count_nonzero(min_3sigma)),
            "colored_events": float(np.count_nonzero(labels)),
        }


def fit_ledoitwolf_metric(
    cache_path: str | Path,
    *,
    max_fit_events: int = 1_000_000,
    random_seed: int = 2026,
    chunk_size: int = 100000,
    workers: int | str | None = "auto",
) -> None:
    try:
        from sklearn.covariance import LedoitWolf
    except Exception as exc:
        raise ImportError("Install the 'ml' extra to fit LedoitWolf covariance") from exc

    mean, scale, finite_count = compute_feature_mean_scale(cache_path, chunk_size=chunk_size)
    with h5py.File(cache_path, "a") as handle:
        features = handle["features"]
        n_events = int(features.shape[0])
        indices = _sample_indices(n_events, max_fit_events, random_seed)
        x_fit = _standardize_block(_read_rows(features, indices, chunk_size=chunk_size), mean, scale)
        with _thread_limit_context(workers):
            model = LedoitWolf().fit(x_fit)
        covariance = np.asarray(model.covariance_, dtype=np.float64)
        precision = np.asarray(model.precision_, dtype=np.float64)

        group = handle.require_group("mahalanobis")
        for key in list(group.keys()):
            del group[key]
        group.attrs["fit_events"] = int(indices.size)
        group.attrs["shrinkage"] = float(model.shrinkage_)
        group.attrs["distance_reference"] = "standardized LedoitWolf feature center"
        group.create_dataset("fit_indices", data=indices)
        group.create_dataset("covariance", data=covariance)
        group.create_dataset("precision", data=precision)
        group.create_dataset("feature_mean", data=mean)
        group.create_dataset("feature_scale", data=scale)
        group.create_dataset("feature_finite_count", data=finite_count)

        if "mahalanobis_distance" in handle:
            del handle["mahalanobis_distance"]
        if "mahalanobis_distance_squared" in handle:
            del handle["mahalanobis_distance_squared"]
        dist = handle.create_dataset("mahalanobis_distance", shape=(n_events,), dtype=np.float32,
                                     chunks=(min(chunk_size, n_events),), compression="lzf")
        dist2 = handle.create_dataset("mahalanobis_distance_squared", shape=(n_events,), dtype=np.float32,
                                      chunks=(min(chunk_size, n_events),), compression="lzf")
        dist.attrs["description"] = "Per-event Mahalanobis distance to the standardized LedoitWolf feature center."
        dist2.attrs["description"] = "Squared per-event Mahalanobis distance to the standardized LedoitWolf feature center."
        with _thread_limit_context(workers):
            for start, stop in _chunk_ranges(n_events, chunk_size):
                x = _standardize_block(features[start:stop], mean, scale)
                projected = x @ precision
                d2 = np.einsum("ij,ij->i", x, projected, optimize=True)
                d2 = np.maximum(d2, 0.0)
                dist2[start:stop] = d2.astype(np.float32)
                dist[start:stop] = np.sqrt(d2).astype(np.float32)


def run_feature_subset_umap_cache(
    cache_path: str | Path,
    *,
    feature_prefixes: Sequence[str],
    group_name: str,
    max_events: int = 200_000,
    random_seed: int = 2026,
    n_neighbors: int = 80,
    min_dist: float = 0.02,
    workers: int | str | None = "auto",
    chunk_size: int = 100000,
) -> None:
    """Run Mahalanobis UMAP on a named subset of cached feature columns."""

    try:
        from sklearn.covariance import LedoitWolf
    except Exception as exc:
        raise ImportError("Install the 'ml' extra to fit subset LedoitWolf covariance") from exc
    try:
        import umap
    except Exception as exc:
        raise ImportError("Install the 'ml' extra to run UMAP") from exc

    feature_mean, feature_scale, _ = _load_or_compute_feature_mean_scale(cache_path, chunk_size=chunk_size)
    cfg = ParallelConfig(workers=workers)
    with h5py.File(cache_path, "a") as handle:
        features = handle["features"]
        n_events = int(features.shape[0])
        feature_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in handle["feature_names"][...]
        ]
        columns, selected_names = _feature_columns_for_prefixes(feature_names, feature_prefixes)
        indices = _sample_indices(n_events, max_events, random_seed)
        mean = feature_mean[columns]
        scale = feature_scale[columns]
        x = _standardize_block(
            _read_column_rows(features, indices, columns, chunk_size=chunk_size),
            mean,
            scale,
        ).astype(np.float32)

        with _thread_limit_context(workers):
            model = LedoitWolf().fit(x)
        covariance = np.asarray(model.covariance_, dtype=np.float64)
        precision = np.asarray(model.precision_, dtype=np.float64)

        reducer = umap.UMAP(
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            n_components=2,
            metric="mahalanobis",
            metric_kwds={"V": covariance},
            init="random",
            random_state=None,
            n_jobs=cfg.resolved_workers(maximum=max(1, x.shape[0])),
            low_memory=True,
        )
        try:
            embedding = reducer.fit_transform(x)
            metric_kwds_used = "V"
        except Exception:
            reducer = umap.UMAP(
                n_neighbors=int(n_neighbors),
                min_dist=float(min_dist),
                n_components=2,
                metric="mahalanobis",
                metric_kwds={"VI": precision},
                init="random",
                random_state=None,
                n_jobs=cfg.resolved_workers(maximum=max(1, x.shape[0])),
                low_memory=True,
            )
            embedding = reducer.fit_transform(x)
            metric_kwds_used = "VI"

        parent = handle
        parts = [part for part in group_name.strip("/").split("/") if part]
        if not parts:
            raise ValueError("group_name must not be empty")
        for part in parts[:-1]:
            parent = parent.require_group(part)
        if parts[-1] in parent:
            del parent[parts[-1]]
        group = parent.create_group(parts[-1])
        string_dtype = h5py.string_dtype("utf-8")
        group.attrs["metric"] = "mahalanobis"
        group.attrs["metric_kwds_used"] = metric_kwds_used
        group.attrs["n_neighbors"] = int(n_neighbors)
        group.attrs["min_dist"] = float(min_dist)
        group.attrs["fit_events"] = int(indices.size)
        group.attrs["shrinkage"] = float(model.shrinkage_)
        group.attrs["feature_prefixes"] = json.dumps(list(feature_prefixes))
        group.create_dataset("feature_columns", data=columns)
        group.create_dataset("feature_names", data=np.asarray(selected_names, dtype=string_dtype))
        group.create_dataset("sample_indices", data=indices)
        group.create_dataset("sample_embedding", data=np.asarray(embedding, dtype=np.float32))
        group.create_dataset("covariance", data=covariance)
        group.create_dataset("precision", data=precision)


def run_umap_cache(
    cache_path: str | Path,
    *,
    max_events: int = 200_000,
    random_seed: int = 2026,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    workers: int | str | None = "auto",
    transform_all: bool = False,
    chunk_size: int = 100000,
    priority_group: str | None = None,
    priority_masks: Sequence[str] | None = None,
    priority_max_events_per_mask: int | None = None,
) -> None:
    try:
        import umap
    except Exception as exc:
        raise ImportError("Install the 'ml' extra to run UMAP") from exc

    cfg = ParallelConfig(workers=workers)
    with h5py.File(cache_path, "a") as handle:
        if "mahalanobis" not in handle:
            raise KeyError("Mahalanobis metric is missing. Run the fit stage first.")
        features = handle["features"]
        n_events = int(features.shape[0])
        metric_group = handle["mahalanobis"]
        mean = np.asarray(metric_group["feature_mean"][...], dtype=np.float64)
        scale = np.asarray(metric_group["feature_scale"][...], dtype=np.float64)
        covariance = np.asarray(metric_group["covariance"][...], dtype=np.float64)
        precision = np.asarray(metric_group["precision"][...], dtype=np.float64)
        priority_indices = np.empty(0, dtype=np.int64)
        if priority_group and priority_masks:
            priority_indices = _priority_indices_from_masks(
                handle,
                priority_group,
                priority_masks,
                chunk_size=chunk_size,
                max_events_per_mask=priority_max_events_per_mask,
                random_seed=random_seed,
            )
        indices = _sample_indices_with_priority(n_events, max_events, random_seed, priority_indices)
        x = _standardize_block(_read_rows(features, indices, chunk_size=chunk_size), mean, scale).astype(np.float32)

        reducer = umap.UMAP(
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            n_components=2,
            metric="mahalanobis",
            metric_kwds={"V": covariance},
            init="random",
            random_state=None,
            n_jobs=cfg.resolved_workers(maximum=max(1, x.shape[0])),
            low_memory=True,
        )
        try:
            embedding = reducer.fit_transform(x)
            metric_kwds_used = "V"
        except Exception:
            reducer = umap.UMAP(
                n_neighbors=int(n_neighbors),
                min_dist=float(min_dist),
                n_components=2,
                metric="mahalanobis",
                metric_kwds={"VI": precision},
                init="random",
                random_state=None,
                n_jobs=cfg.resolved_workers(maximum=max(1, x.shape[0])),
                low_memory=True,
            )
            embedding = reducer.fit_transform(x)
            metric_kwds_used = "VI"

        group = handle.require_group("umap")
        for key in ("sample_indices", "sample_embedding", "all_embedding"):
            if key in group:
                del group[key]
        group.attrs["metric"] = "mahalanobis"
        group.attrs["metric_kwds_used"] = metric_kwds_used
        group.attrs["n_neighbors"] = int(n_neighbors)
        group.attrs["min_dist"] = float(min_dist)
        group.attrs["priority_group"] = "" if priority_group is None else str(priority_group)
        group.attrs["priority_masks"] = json.dumps(list(priority_masks or []))
        group.attrs["priority_events_available"] = int(priority_indices.size)
        group.attrs["priority_events_in_sample"] = int(np.intersect1d(indices, priority_indices, assume_unique=True).size)
        group.attrs["priority_max_events_per_mask"] = (
            -1 if priority_max_events_per_mask is None else int(priority_max_events_per_mask)
        )
        group.create_dataset("sample_indices", data=indices)
        group.create_dataset("sample_embedding", data=np.asarray(embedding, dtype=np.float32))

        if transform_all:
            all_embedding = group.create_dataset(
                "all_embedding",
                shape=(n_events, 2),
                dtype=np.float32,
                chunks=(min(chunk_size, n_events), 2),
                compression="lzf",
            )
            for start, stop in _chunk_ranges(n_events, chunk_size):
                block = _standardize_block(features[start:stop], mean, scale).astype(np.float32)
                all_embedding[start:stop] = reducer.transform(block).astype(np.float32)


def _mask_values_for_indices(handle: h5py.File, mask_name: str, indices: np.ndarray) -> np.ndarray:
    return _read_vector(handle[f"masks/{mask_name}"], indices)


def _group_values_for_indices(handle: h5py.File, group_name: str, dataset_name: str, indices: np.ndarray) -> np.ndarray:
    return _read_vector(handle[f"{group_name.strip('/')}/{dataset_name}"], indices)


def plot_umap_masks(
    cache_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    point_size: float = 3.0,
    embedding_group: str = "umap",
    title: str = "UMAP colored by physics masks",
    filename_prefix: str = "umap",
) -> list[Path]:
    import matplotlib.pyplot as plt

    cache = Path(cache_path)
    out_dir = Path(output_dir) if output_dir is not None else cache.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with h5py.File(cache, "r") as handle:
        group = handle[embedding_group.strip("/")]
        embedding = np.asarray(group["sample_embedding"][...], dtype=np.float32)
        indices = np.asarray(group["sample_indices"][...], dtype=np.int64)
        labels = _mask_values_for_indices(handle, "event_class", indices).astype(np.int16)
        label_names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in handle["masks/event_class_names"][...]
        ]
        colors = {
            0: "#7A7A7A",
            1: "#D55E00",
            2: "#CC79A7",
            3: "#CC79A7",
            4: "#C44E52",
            5: "#0072B2",
            6: "#E69F00",
            7: "#009E73",
        }

        fig, ax = plt.subplots(figsize=(8, 6))
        for code in sorted(np.unique(labels)):
            m = labels == code
            ax.scatter(
                embedding[m, 0],
                embedding[m, 1],
                s=point_size,
                alpha=0.65,
                c=colors.get(int(code), "#333333"),
                label=label_names[int(code)] if int(code) < len(label_names) else str(code),
                linewidths=0,
            )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(markerscale=3, fontsize=9)
        ax.set_title(title)
        fig.tight_layout()
        path = out_dir / f"{filename_prefix}_by_event_class.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)

        panel_masks = ["rt", "inhibit", "pedestal_outlier", "over_threshold", "act", "clean"]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
        for ax, name in zip(axes.ravel(), panel_masks):
            mask = _mask_values_for_indices(handle, name, indices).astype(bool)
            ax.scatter(embedding[:, 0], embedding[:, 1], s=1.0, alpha=0.18, c="#B0B0B0", linewidths=0)
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=point_size, alpha=0.75, linewidths=0)
            ax.set_title(f"{name}: {int(mask.sum())}")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        path = out_dir / f"{filename_prefix}_mask_panels.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_anomaly_umap(
    cache_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    point_size: float = 3.0,
    embedding_group: str = "umap",
    anomaly_group: str = "anomaly_masks",
    title: str = "UMAP with physics anomaly classes highlighted",
    filename_prefix: str = "anomaly_umap",
) -> list[Path]:
    import matplotlib.pyplot as plt

    cache = Path(cache_path)
    out_dir = Path(output_dir) if output_dir is not None else cache.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with h5py.File(cache, "r") as handle:
        group = handle[embedding_group.strip("/")]
        embedding = np.asarray(group["sample_embedding"][...], dtype=np.float32)
        indices = np.asarray(group["sample_indices"][...], dtype=np.int64)
        labels = _group_values_for_indices(handle, anomaly_group, "event_class", indices).astype(np.int16)
        label_names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in handle[f"{anomaly_group.strip('/')}/event_class_names"][...]
        ]
        colors = {
            1: "#CC79A7",
            2: "#D55E00",
            3: "#CC79A7",
            4: "#C44E52",
            5: "#0072B2",
        }

        fig, ax = plt.subplots(figsize=(8, 6))
        background = labels == 0
        ax.scatter(
            embedding[background, 0],
            embedding[background, 1],
            s=1.0,
            alpha=0.12,
            c="#A8A8A8",
            label=label_names[0],
            linewidths=0,
        )
        for code in sorted(c for c in np.unique(labels) if int(c) != 0):
            mask = labels == code
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=point_size,
                alpha=0.78,
                c=colors.get(int(code), "#333333"),
                label=f"{label_names[int(code)]}: {int(mask.sum())}",
                linewidths=0,
            )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(markerscale=3, fontsize=9)
        ax.set_title(title)
        fig.tight_layout()
        path = out_dir / f"{filename_prefix}_highlighted.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)

        panel_masks = [
            "inhibit",
            "random_trigger",
            "pedestal_3sigma_outlier",
            "over_threshold",
            "act",
            "min_3sigma_outlier",
        ]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
        for ax, name in zip(axes.ravel(), panel_masks):
            mask = _group_values_for_indices(handle, anomaly_group, name, indices).astype(bool)
            ax.scatter(embedding[:, 0], embedding[:, 1], s=1.0, alpha=0.12, c="#A8A8A8", linewidths=0)
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=point_size,
                alpha=0.78,
                c="#0072B2",
                linewidths=0,
            )
            ax.set_title(f"{name}: {int(mask.sum())}")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        path = out_dir / f"{filename_prefix}_panels.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


__all__ = [
    "DEFAULT_CACHE_NAME",
    "build_feature_cache",
    "compute_event_masks",
    "compute_feature_mean_scale",
    "discover_parameter_channels",
    "fit_ledoitwolf_metric",
    "load_feature_names",
    "paired_parameter_runs",
    "plot_umap_masks",
    "compute_basic_anomaly_masks",
    "plot_anomaly_umap",
    "run_feature_subset_umap_cache",
    "run_umap_cache",
]
