#!/usr/bin/env python
"""Plot representative or random CH0-3 waveforms for HDBSCAN clusters."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.features import time_axis  # noqa: E402
from analysis.io.paths import find_project_root, raw_pulse_dir  # noqa: E402


def _default_cluster_path(root: Path) -> Path:
    return (
        root
        / "data"
        / "cache"
        / "hdbscan_clean_remaining_umap_remote_8clusters"
        / "hdbscan_umap_clusters.h5"
    )


def _default_feature_cache(root: Path) -> Path:
    return root / "data" / "cache" / "event_feature_umap" / "event_feature_cache.h5"


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _short_run_name(name: str) -> str:
    match = re.search(r"RAW_Data_(\d+)_processed", name)
    if match:
        return f"RAW_Data_{match.group(1)}"
    return Path(name).stem[:42]


def _read_indexed(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary 1D positions from an HDF5 dataset with sorted point access."""

    idx = np.asarray(indices, dtype=np.int64)
    order = np.argsort(idx)
    sorted_idx = idx[order]
    values = np.asarray(dataset[sorted_idx])
    out = np.empty_like(values)
    out[order] = values
    return out


def _load_sample_metadata(
    feature_cache: Path,
    sample_indices: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(feature_cache, "r") as handle:
        run_names = [_decode(x) for x in handle["run_names"][...]]
        # These arrays are small compared with the feature matrix and much faster
        # to read once than to point-select 200k sampled events from HDF5.
        run_index = np.asarray(handle["run_index"][...], dtype=np.int64)[sample_indices]
        event_index = np.asarray(handle["event_index"][...], dtype=np.int64)[sample_indices]
        event_time = np.asarray(handle["event_time"][...], dtype=np.float64)[sample_indices]
    return run_names, run_index, event_index, event_time


def _best_compact_window(
    positions: np.ndarray,
    *,
    take: int,
    sample_event_index: np.ndarray,
    distance: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    if positions.size < take:
        raise ValueError("positions must contain at least take entries")
    order = np.argsort(sample_event_index[positions], kind="mergesort")
    ordered = positions[order]
    events = sample_event_index[ordered]
    best_positions = ordered[:take]
    best_score = (float("inf"), float("inf"), float("inf"))
    for start in range(0, ordered.size - take + 1):
        stop = start + take
        window = ordered[start:stop]
        span = int(events[stop - 1] - events[start])
        mean_distance = float(np.mean(distance[window]))
        # Negative probability lets lexicographic min prefer confident events.
        neg_probability = -float(np.mean(probabilities[window]))
        score = (float(span), mean_distance, neg_probability)
        if score < best_score:
            best_score = score
            best_positions = window
    return np.asarray(best_positions, dtype=np.int64), best_score


def _select_label_positions(
    *,
    label: int,
    positions: np.ndarray,
    embedding: np.ndarray,
    probabilities: np.ndarray,
    sample_run_index: np.ndarray,
    sample_event_index: np.ndarray,
    events_per_cluster: int,
    rng: np.random.Generator,
    selection: str,
) -> tuple[np.ndarray, str]:
    n_take = min(int(events_per_cluster), int(positions.size))
    if n_take <= 0:
        return np.empty(0, dtype=np.int64), "empty"
    if selection == "random":
        selected = rng.choice(positions, size=n_take, replace=False)
        return np.asarray(selected, dtype=np.int64), "random"
    if selection == "random-local":
        shuffled = rng.permutation(positions)
        for anchor in shuffled:
            run_idx = int(sample_run_index[anchor])
            run_positions = positions[sample_run_index[positions] == run_idx]
            if run_positions.size < n_take:
                continue
            event_distance = np.abs(sample_event_index[run_positions] - sample_event_index[anchor])
            tie_breaker = rng.random(run_positions.size)
            order = np.lexsort((tie_breaker, event_distance))
            return np.asarray(run_positions[order[:n_take]], dtype=np.int64), "random_local"
        selected = rng.choice(positions, size=n_take, replace=False)
        return np.asarray(selected, dtype=np.int64), "random_fallback"

    if label < 0:
        distance = np.zeros(embedding.shape[0], dtype=np.float64)
    else:
        center = np.median(embedding[positions], axis=0)
        distance = np.linalg.norm(embedding - center, axis=1)

    by_run: dict[int, np.ndarray] = {}
    for run_idx in np.unique(sample_run_index[positions]):
        run_positions = positions[sample_run_index[positions] == run_idx]
        by_run[int(run_idx)] = run_positions

    exact_candidates: list[tuple[tuple[float, float, float], np.ndarray]] = []
    for run_positions in by_run.values():
        if run_positions.size < n_take:
            continue
        window, score = _best_compact_window(
            run_positions,
            take=n_take,
            sample_event_index=sample_event_index,
            distance=distance,
            probabilities=probabilities,
        )
        exact_candidates.append((score, window))

    if exact_candidates:
        exact_candidates.sort(key=lambda item: item[0])
        return exact_candidates[0][1], "single_run_compact"

    if label < 0:
        selected = rng.choice(positions, size=n_take, replace=False)
        return np.asarray(selected, dtype=np.int64), "noise_random"

    selected_parts: list[np.ndarray] = []
    remaining = n_take
    run_candidates: list[tuple[int, tuple[float, float, float], np.ndarray]] = []
    for run_positions in by_run.values():
        take = min(remaining, int(run_positions.size))
        if take <= 0:
            continue
        window, score = _best_compact_window(
            run_positions,
            take=take,
            sample_event_index=sample_event_index,
            distance=distance,
            probabilities=probabilities,
        )
        run_candidates.append((int(run_positions.size), score, window))
    run_candidates.sort(key=lambda item: (-item[0], item[1]))

    for _, _, window in run_candidates:
        if remaining <= 0:
            break
        take = min(remaining, window.size)
        selected_parts.append(window[:take])
        remaining -= take

    if remaining > 0:
        order = np.lexsort((-probabilities[positions], distance[positions]))
        selected_parts.append(positions[order[:remaining]])

    return np.concatenate(selected_parts).astype(np.int64), "multi_run_compact"


def _select_representatives(
    *,
    labels: np.ndarray,
    embedding: np.ndarray,
    probabilities: np.ndarray,
    sample_indices: np.ndarray,
    sample_run_index: np.ndarray,
    sample_event_index: np.ndarray,
    sample_event_time: np.ndarray,
    run_names: list[str],
    events_per_cluster: int,
    random_seed: int,
    selection: str,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, object]] = []
    for label in sorted(int(x) for x in np.unique(labels)):
        mask = labels == label
        positions = np.flatnonzero(mask)
        if positions.size == 0:
            continue
        selected, mode = _select_label_positions(
            label=label,
            positions=positions,
            embedding=embedding,
            probabilities=probabilities,
            sample_run_index=sample_run_index,
            sample_event_index=sample_event_index,
            events_per_cluster=events_per_cluster,
            rng=rng,
            selection=selection,
        )
        if selection != "random":
            selected = selected[np.lexsort((sample_event_index[selected], sample_run_index[selected]))]

        for rank, pos in enumerate(selected, start=1):
            run_idx = int(sample_run_index[pos])
            rows.append(
                {
                    "cluster": int(label),
                    "cluster_name": "noise" if label < 0 else f"C{label}",
                    "rank": int(rank),
                    "selection_mode": mode,
                    "hdbscan_row": int(pos),
                    "global_index": int(sample_indices[pos]),
                    "run_index": run_idx,
                    "run_name": run_names[run_idx],
                    "event_index": int(sample_event_index[pos]),
                    "event_time": float(sample_event_time[pos]),
                    "umap1": float(embedding[pos, 0]),
                    "umap2": float(embedding[pos, 1]),
                    "probability": float(probabilities[pos]),
                }
            )
    return rows


def _attach_run_event_metadata(rows: list[dict[str, object]], feature_cache: Path) -> None:
    global_indices = np.asarray([row["global_index"] for row in rows], dtype=np.int64)
    with h5py.File(feature_cache, "r") as handle:
        run_names = [_decode(x) for x in handle["run_names"][...]]
        run_index = _read_indexed(handle["run_index"], global_indices).astype(np.int64)
        event_index = _read_indexed(handle["event_index"], global_indices).astype(np.int64)
        event_time = _read_indexed(handle["event_time"], global_indices).astype(np.float64)

    for row, run_idx, event_idx, timestamp in zip(rows, run_index, event_index, event_time):
        row["run_index"] = int(run_idx)
        row["run_name"] = run_names[int(run_idx)]
        row["event_index"] = int(event_idx)
        row["event_time"] = float(timestamp)


def _read_waveforms_by_run(
    rows: list[dict[str, object]],
    project_root: Path,
    *,
    channel_indices: list[int],
    max_event_gap: int,
) -> dict[tuple[int, int], dict[int, np.ndarray]]:
    by_run: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_run[str(row["run_name"])].append(row)

    waveforms: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    ch03_dir = raw_pulse_dir(project_root) / "CH0-3"
    max_channel = max(channel_indices)
    for run_name, run_rows in by_run.items():
        path = ch03_dir / run_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw CH0-3 file for {run_name}: {path}")
        with h5py.File(path, "r") as handle:
            if "channel_data" not in handle:
                raise KeyError(f"{path} does not contain channel_data")
            data = handle["channel_data"]
            if data.ndim != 3:
                raise ValueError(f"{path}: channel_data must be (time, channel, event), got {data.shape}")
            _, n_channels, n_events = data.shape
            if n_channels <= max_channel:
                raise ValueError(
                    f"{path}: channel_data has only {n_channels} channels; requested CH{max_channel}"
                )
            ordered_rows = sorted(run_rows, key=lambda item: int(item["event_index"]))
            segments: list[list[dict[str, object]]] = []
            current: list[dict[str, object]] = []
            last_event: int | None = None
            for row in ordered_rows:
                event_idx = int(row["event_index"])
                if event_idx < 0 or event_idx >= n_events:
                    raise IndexError(f"{path}: event_index={event_idx} outside [0, {n_events - 1}]")
                if current and last_event is not None and event_idx - last_event > int(max_event_gap):
                    segments.append(current)
                    current = []
                current.append(row)
                last_event = event_idx
            if current:
                segments.append(current)

            for segment in segments:
                start = min(int(row["event_index"]) for row in segment)
                stop = max(int(row["event_index"]) for row in segment) + 1
                block = np.asarray(data[:, :, start:stop], dtype=np.float64)
                for row in segment:
                    event_idx = int(row["event_index"])
                    key = (int(row["cluster"]), int(row["rank"]))
                    waveforms[key] = {
                        channel: np.asarray(block[:, channel, event_idx - start], dtype=np.float64)
                        for channel in channel_indices
                    }
    return waveforms


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "cluster",
            "cluster_name",
            "rank",
            "selection_mode",
            "hdbscan_row",
            "global_index",
            "run_index",
            "run_name",
            "event_index",
            "event_time",
            "umap1",
            "umap2",
            "probability",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_one_cluster(
    *,
    output_dir: Path,
    cluster_rows: list[dict[str, object]],
    waveforms: dict[tuple[int, int], dict[int, np.ndarray]],
    channel_indices: list[int],
    sampling_interval_ns: float,
    line_width: float,
    dpi: int,
) -> Path:
    import matplotlib.pyplot as plt

    if not cluster_rows:
        raise ValueError("cluster_rows is empty")
    cluster = int(cluster_rows[0]["cluster"])
    name = str(cluster_rows[0]["cluster_name"])
    n_rows = len(cluster_rows)
    n_cols = len(channel_indices)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(13.5, 4.4 * n_cols), max(7.0, 1.8 * n_rows)),
        sharex=False,
    )
    if n_rows == 1:
        axes = np.asarray([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    colors = ["#0072B2", "#009E73", "#CC79A7", "#D55E00", "#E69F00", "#56B4E9"]

    for row_idx, row in enumerate(cluster_rows):
        key = (int(row["cluster"]), int(row["rank"]))
        label = (
            f"{row['cluster_name']} #{row['rank']} | "
            f"{_short_run_name(str(row['run_name']))} ev={row['event_index']} "
            f"p={float(row['probability']):.3f}"
        )

        for col_idx, channel in enumerate(channel_indices):
            ax = axes[row_idx, col_idx]
            waveform = waveforms[key][channel]
            x = time_axis(waveform.size, sampling_interval_ns=sampling_interval_ns, unit="us")
            ax.plot(x, waveform, color=colors[col_idx % len(colors)], linewidth=line_width)
            if col_idx == 0:
                ax.set_ylabel("ADC")
                ax.set_title(f"{label}\nCH{channel}", fontsize=8.5, loc="left")
            else:
                ax.set_title(f"CH{channel}", fontsize=8.5, loc="left")
            ax.tick_params(labelsize=7)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time (us)")

    channel_title = ", ".join(f"CH{channel}" for channel in channel_indices)
    fig.suptitle(f"HDBSCAN {name} waveforms: {channel_title}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if channel_indices == [0, 1, 2, 3]:
        suffix = "ch0_3"
    else:
        suffix = "_".join(f"ch{channel}" for channel in channel_indices)
    filename = f"cluster_noise_waveforms_{suffix}.png" if cluster < 0 else f"cluster_{cluster:02d}_waveforms_{suffix}.png"
    path = output_dir / filename
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CH0-3 waveform examples for each HDBSCAN cluster.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--cluster-h5", default=None, help="HDBSCAN h5 with labels/sample_indices/embedding.")
    parser.add_argument("--feature-cache", default=None, help="Event feature cache with run_index/event_index.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--events-per-cluster", type=int, default=9)
    parser.add_argument(
        "--selection",
        choices=("compact", "random", "random-local"),
        default="compact",
        help=(
            "compact: representative local window; random: global random cluster members; "
            "random-local: random anchor plus nearby same-run cluster members."
        ),
    )
    parser.add_argument("--channels", nargs="+", type=int, default=[0, 3], help="CH0-3 channel indices to plot.")
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--sampling-interval-ns", type=float, default=4.0)
    parser.add_argument("--max-event-gap", type=int, default=128)
    parser.add_argument("--line-width", type=float, default=0.65)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    root = find_project_root(args.project_root)
    cluster_h5 = Path(args.cluster_h5) if args.cluster_h5 else _default_cluster_path(root)
    feature_cache = Path(args.feature_cache) if args.feature_cache else _default_feature_cache(root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "data" / "cache" / "hdbscan_cluster_waveforms_8clusters"
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_indices = sorted(dict.fromkeys(int(channel) for channel in args.channels))
    if not channel_indices or min(channel_indices) < 0 or max(channel_indices) > 3:
        raise ValueError("--channels must be one or more CH0-3 indices in [0, 3]")

    with h5py.File(cluster_h5, "r") as handle:
        labels = np.asarray(handle["labels"][...], dtype=np.int64)
        embedding = np.asarray(handle["embedding"][...], dtype=np.float64)
        sample_indices = np.asarray(handle["sample_indices"][...], dtype=np.int64)
        probabilities = np.asarray(handle["probabilities"][...], dtype=np.float64)

    run_names, sample_run_index, sample_event_index, sample_event_time = _load_sample_metadata(
        feature_cache,
        sample_indices,
    )
    rows = _select_representatives(
        labels=labels,
        embedding=embedding,
        probabilities=probabilities,
        sample_indices=sample_indices,
        sample_run_index=sample_run_index,
        sample_event_index=sample_event_index,
        sample_event_time=sample_event_time,
        run_names=run_names,
        events_per_cluster=int(args.events_per_cluster),
        random_seed=int(args.random_seed),
        selection=str(args.selection),
    )
    waveforms = _read_waveforms_by_run(
        rows,
        root,
        channel_indices=channel_indices,
        max_event_gap=int(args.max_event_gap),
    )

    rows_by_cluster: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_cluster[int(row["cluster"])].append(row)

    written: list[Path] = []
    for cluster in sorted(rows_by_cluster):
        cluster_rows = sorted(rows_by_cluster[cluster], key=lambda item: int(item["rank"]))
        path = _plot_one_cluster(
            output_dir=output_dir,
            cluster_rows=cluster_rows,
            waveforms=waveforms,
            channel_indices=channel_indices,
            sampling_interval_ns=float(args.sampling_interval_ns),
            line_width=float(args.line_width),
            dpi=int(args.dpi),
        )
        written.append(path)
        print(f"Wrote {path}", flush=True)

    csv_path = output_dir / "selected_cluster_waveform_events.csv"
    _write_rows(csv_path, rows)
    print(f"Wrote {csv_path}", flush=True)
    print(f"figures: {len(written)}", flush=True)


if __name__ == "__main__":
    main()
