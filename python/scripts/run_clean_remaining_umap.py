#!/usr/bin/env python
"""Run all-feature UMAP on events remaining after basic anomaly removals."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.io.paths import find_project_root  # noqa: E402
from analysis.ml.event_matrix import DEFAULT_CACHE_NAME, compute_basic_anomaly_masks, run_umap_cache  # noqa: E402
from analysis.parallel import add_parallel_arguments  # noqa: E402


DEFAULT_EXCLUDE_MASKS = (
    "fit_failed",
    "inhibit",
    "random_trigger",
    "min_3sigma_outlier",
    "pedestal_3sigma_outlier",
    "act",
    "over_threshold",
)


def _cache_path(args: argparse.Namespace) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir) / DEFAULT_CACHE_NAME
    root = find_project_root(args.project_root)
    return root / "data" / "cache" / "event_feature_umap" / DEFAULT_CACHE_NAME


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    root = find_project_root(args.project_root)
    return root / "data" / "cache" / "clean_remaining_umap"


def _chunk_ranges(n_items: int, chunk_size: int):
    for start in range(0, n_items, chunk_size):
        yield start, min(start + chunk_size, n_items)


def _read_vector_for_indices(dataset: h5py.Dataset, indices: np.ndarray, *, chunk_size: int = 100000) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    order = np.argsort(idx)
    sorted_idx = idx[order]
    out = np.empty(idx.size, dtype=dataset.dtype)
    n_events = int(dataset.shape[0])
    for start, stop in _chunk_ranges(n_events, int(chunk_size)):
        left = int(np.searchsorted(sorted_idx, start, side="left"))
        right = int(np.searchsorted(sorted_idx, stop, side="left"))
        if right <= left:
            continue
        block = np.asarray(dataset[start:stop])
        out[order[left:right]] = block[sorted_idx[left:right] - start]
    return out


def build_clean_remaining_mask(
    cache_path: str | Path,
    *,
    anomaly_group: str = "anomaly_masks",
    clean_mask_name: str = "clean_remaining",
    exclude_masks: Sequence[str] = DEFAULT_EXCLUDE_MASKS,
    chunk_size: int = 100000,
) -> dict[str, int]:
    """Create a clean-remaining mask by excluding the requested anomaly masks."""

    cache = Path(cache_path)
    with h5py.File(cache, "a") as handle:
        group = handle[anomaly_group.strip("/")]
        missing = [name for name in exclude_masks if name not in group]
        if missing:
            raise KeyError(f"Missing anomaly mask(s) under {anomaly_group}: {missing}")

        n_events = int(group[exclude_masks[0]].shape[0])
        if clean_mask_name in group:
            del group[clean_mask_name]
        clean_dset = group.create_dataset(
            clean_mask_name,
            shape=(n_events,),
            dtype=np.uint8,
            chunks=(min(int(chunk_size), n_events),),
            compression="lzf",
        )

        excluded_counts = {name: 0 for name in exclude_masks}
        clean_count = 0
        excluded_union_count = 0
        for start, stop in _chunk_ranges(n_events, int(chunk_size)):
            excluded = np.zeros(stop - start, dtype=bool)
            for name in exclude_masks:
                mask = np.asarray(group[name][start:stop], dtype=bool)
                excluded_counts[name] += int(np.count_nonzero(mask))
                excluded |= mask
            clean = ~excluded
            clean_dset[start:stop] = clean.astype(np.uint8)
            clean_count += int(np.count_nonzero(clean))
            excluded_union_count += int(np.count_nonzero(excluded))

        clean_dset.attrs["definition"] = "not any excluded anomaly mask"
        clean_dset.attrs["exclude_masks"] = ",".join(exclude_masks)
        clean_dset.attrs["clean_count"] = int(clean_count)
        clean_dset.attrs["excluded_union_count"] = int(excluded_union_count)

    return {
        "total_events": int(n_events),
        "clean_remaining": int(clean_count),
        "excluded_union": int(excluded_union_count),
        **{f"excluded_{key}": int(value) for key, value in excluded_counts.items()},
    }


def plot_clean_remaining_umap(
    cache_path: str | Path,
    *,
    output_dir: str | Path,
    title: str,
    anomaly_group: str = "anomaly_masks",
    clean_mask_name: str = "clean_remaining",
    embedding_group: str = "umap",
    point_size: float = 2.0,
) -> Path:
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_path, "r") as handle:
        group = handle[embedding_group.strip("/")]
        embedding = np.asarray(group["sample_embedding"][...], dtype=np.float32)
        indices = np.asarray(group["sample_indices"][...], dtype=np.int64)
        clean_values = np.asarray(
            _read_vector_for_indices(handle[f"{anomaly_group.strip('/')}/{clean_mask_name}"], indices),
            dtype=bool,
        )
        n_clean = int(np.count_nonzero(clean_values))
        n_sample = int(indices.size)
        available = int(handle[f"{anomaly_group.strip('/')}/{clean_mask_name}"].attrs.get("clean_count", n_clean))

    fig, ax = plt.subplots(figsize=(8, 6))
    if np.any(~clean_values):
        ax.scatter(
            embedding[~clean_values, 0],
            embedding[~clean_values, 1],
            s=point_size,
            alpha=0.55,
            c="#C44E52",
            linewidths=0,
            label=f"unexpected excluded: {int(np.count_nonzero(~clean_values))}",
        )
    ax.scatter(
        embedding[clean_values, 0],
        embedding[clean_values, 1],
        s=point_size,
        alpha=0.45,
        c="#0072B2",
        linewidths=0,
        label=f"clean remaining sample: {n_clean}/{n_sample}",
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"{title}\navailable clean remaining={available}, sample={n_sample}")
    ax.legend(markerscale=3, fontsize=9)
    fig.tight_layout()
    path = out_dir / "clean_remaining_umap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove inhibit, RT/random-trigger, min-3sigma, pedestal-3sigma, ACT, "
            "fit-failed, and over-threshold events, then run an all-feature UMAP on the remaining events."
        )
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--cache-dir", default=None, help="Default: data/cache/event_feature_umap")
    parser.add_argument("--output-dir", default=None, help="Default: data/cache/clean_remaining_umap")
    parser.add_argument("--umap-max-events", type=int, default=200_000, help="0 means all clean remaining events.")
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--umap-neighbors", type=int, default=400)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--rt-threshold", type=float, default=6000.0)
    parser.add_argument("--max-adc", type=float, default=16382.0)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--skip-masks", action="store_true", help="Do not recompute anomaly masks before running.")
    parser.add_argument("--clean-mask-name", default="clean_remaining")
    parser.add_argument("--exclude-masks", nargs="+", default=list(DEFAULT_EXCLUDE_MASKS))
    add_parallel_arguments(parser, include_chunk_size=True, default_chunk_size=100000)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cache_path = _cache_path(args)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache does not exist: {cache_path}")
    out_dir = _output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    if not args.skip_masks:
        stats = compute_basic_anomaly_masks(
            cache_path,
            rt_threshold=args.rt_threshold,
            max_adc=args.max_adc,
            n_sigma=args.sigma,
            chunk_size=args.chunk_size,
        )
        for key, value in stats.items():
            print(f"{key}: {int(value)}", flush=True)
            rows.append({"metric": key, "value": int(value)})

    clean_stats = build_clean_remaining_mask(
        cache_path,
        clean_mask_name=args.clean_mask_name,
        exclude_masks=tuple(args.exclude_masks),
        chunk_size=args.chunk_size,
    )
    for key, value in clean_stats.items():
        print(f"{key}: {int(value)}", flush=True)
        rows.append({"metric": key, "value": int(value)})

    clean_count = int(clean_stats["clean_remaining"])
    if clean_count <= 0:
        raise RuntimeError("No clean remaining events are available after exclusions.")
    umap_max_events = clean_count if int(args.umap_max_events) <= 0 else min(int(args.umap_max_events), clean_count)
    print(f"umap_requested_sample_events: {umap_max_events}", flush=True)
    rows.append({"metric": "umap_requested_sample_events", "value": int(umap_max_events)})

    run_umap_cache(
        cache_path,
        max_events=umap_max_events,
        random_seed=args.random_seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        workers=args.workers,
        chunk_size=args.chunk_size,
        priority_group="anomaly_masks",
        priority_masks=(args.clean_mask_name,),
        priority_max_events_per_mask=None,
    )

    with h5py.File(cache_path, "r") as handle:
        group = handle["umap"]
        rows.extend(
            [
                {"metric": "umap_n_neighbors", "value": int(group.attrs["n_neighbors"])},
                {"metric": "umap_min_dist", "value": float(group.attrs["min_dist"])},
                {"metric": "umap_sample_events", "value": int(group["sample_indices"].shape[0])},
                {"metric": "clean_events_in_sample", "value": int(group.attrs["priority_events_in_sample"])},
            ]
        )

    png = plot_clean_remaining_umap(
        cache_path,
        output_dir=out_dir,
        title=(
            "Clean remaining all-feature UMAP\n"
            "removed: fit-failed, inhibit, RT, ACT, min/pedestal 3sigma, over-threshold\n"
            f"n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist}"
        ),
        clean_mask_name=args.clean_mask_name,
        point_size=args.point_size,
    )
    _write_summary(out_dir / "clean_remaining_umap_summary.csv", rows)
    print(f"Wrote {png}", flush=True)
    print(f"Wrote {out_dir / 'clean_remaining_umap_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
