#!/usr/bin/env python
"""Cluster a cached UMAP embedding with an adaptive HDBSCAN parameter scan."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.io.paths import find_project_root  # noqa: E402
from analysis.ml.event_matrix import DEFAULT_CACHE_NAME  # noqa: E402


def _cache_path(args: argparse.Namespace) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir) / DEFAULT_CACHE_NAME
    root = find_project_root(args.project_root)
    return root / "data" / "cache" / "event_feature_umap" / DEFAULT_CACHE_NAME


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    root = find_project_root(args.project_root)
    return root / "data" / "cache" / "hdbscan_umap_clusters"


def _parse_ints(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(part) for part in text.replace(",", " ").split()]


def _auto_min_cluster_sizes(n_events: int) -> list[int]:
    values: set[int] = set()
    for frac in (0.0025, 0.005, 0.01, 0.025, 0.05, 0.10):
        values.add(max(20, int(round(n_events * frac))))
    for fixed in (500, 1000, 2000, 5000, 10000, 20000):
        if fixed < n_events:
            values.add(fixed)
    return sorted(values)


def _auto_min_samples(min_cluster_sizes: Iterable[int]) -> list[int | None]:
    values: set[int | None] = {None, 10, 25, 50, 100, 200}
    for size in min_cluster_sizes:
        values.add(max(10, int(size // 20)))
    return sorted(values, key=lambda x: -1 if x is None else int(x))


def _cluster_stats(labels: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    n = int(labels.size)
    clustered = labels >= 0
    clusters = np.unique(labels[clustered])
    counts = np.array([np.count_nonzero(labels == label) for label in clusters], dtype=np.int64)
    noise = int(np.count_nonzero(~clustered))
    if counts.size:
        largest = int(counts.max())
        smallest = int(counts.min())
        median = float(np.median(counts))
    else:
        largest = 0
        smallest = 0
        median = 0.0
    mean_probability = float(np.mean(probabilities[clustered])) if probabilities is not None and np.any(clustered) else 0.0
    return {
        "n_events": float(n),
        "n_clusters": float(clusters.size),
        "noise_events": float(noise),
        "noise_fraction": float(noise / n) if n else 1.0,
        "clustered_fraction": float(np.count_nonzero(clustered) / n) if n else 0.0,
        "largest_cluster_events": float(largest),
        "largest_cluster_fraction": float(largest / n) if n else 0.0,
        "smallest_cluster_events": float(smallest),
        "median_cluster_events": median,
        "mean_probability": mean_probability,
    }


def _score_candidate(stats: dict[str, float], target_min: int, target_max: int) -> float:
    n_clusters = int(stats["n_clusters"])
    if n_clusters <= 1:
        return -1e9
    noise = float(stats["noise_fraction"])
    largest = float(stats["largest_cluster_fraction"])
    prob = float(stats["mean_probability"])

    if target_min <= n_clusters <= target_max:
        cluster_score = 1.0
    else:
        distance = min(abs(n_clusters - target_min), abs(n_clusters - target_max))
        cluster_score = max(0.0, 1.0 - 0.08 * distance)

    # Prefer readable island-level clustering: modest noise, no single cluster swallowing everything,
    # and reasonably confident HDBSCAN assignments.
    noise_score = max(0.0, 1.0 - abs(noise - 0.08) / 0.35)
    balance_score = max(0.0, 1.0 - max(0.0, largest - 0.65) / 0.35)
    return 4.0 * cluster_score + 2.0 * noise_score + 1.5 * balance_score + 1.5 * prob


def _fit_hdbscan(embedding: np.ndarray, min_cluster_size: int, min_samples: int | None):
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(np.asarray(embedding, dtype=np.float64))
    probabilities = np.asarray(getattr(clusterer, "probabilities_", np.ones(labels.size)), dtype=np.float64)
    return labels, probabilities, clusterer


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_clusters(
    path: Path,
    embedding: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    title: str,
    point_size: float,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    labels = np.asarray(labels, dtype=np.int64)
    clusters = sorted(int(x) for x in np.unique(labels) if int(x) >= 0)
    noise = labels < 0
    if np.any(noise):
        ax.scatter(
            embedding[noise, 0],
            embedding[noise, 1],
            s=max(0.5, point_size * 0.65),
            c="#B8B8B8",
            alpha=0.18,
            linewidths=0,
            label=f"noise: {int(np.count_nonzero(noise))}",
        )
    cmap = plt.get_cmap("tab20")
    for idx, label in enumerate(clusters):
        mask = labels == label
        alpha = np.clip(0.35 + 0.55 * probabilities[mask], 0.35, 0.9)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            c=[cmap(idx % cmap.N)],
            alpha=alpha,
            linewidths=0,
            label=f"C{label}: {int(np.count_nonzero(mask))}",
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    if len(clusters) <= 20:
        ax.legend(markerscale=3, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive HDBSCAN clustering on a cached UMAP embedding.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--cache-dir", default=None, help="Default: data/cache/event_feature_umap")
    parser.add_argument("--output-dir", default=None, help="Default: data/cache/hdbscan_umap_clusters")
    parser.add_argument("--embedding-group", default="umap")
    parser.add_argument("--min-cluster-sizes", default=None, help="Comma/space separated values. Default: auto.")
    parser.add_argument("--min-samples", default=None, help="Comma/space separated values. Default: auto plus None.")
    parser.add_argument("--target-min-clusters", type=int, default=3)
    parser.add_argument("--target-max-clusters", type=int, default=18)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--max-grid", type=int, default=120, help="Cap total candidate fits.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cache_path = _cache_path(args)
    out_dir = _output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(cache_path, "r") as handle:
        group = handle[args.embedding_group.strip("/")]
        embedding = np.asarray(group["sample_embedding"][...], dtype=np.float32)
        sample_indices = np.asarray(group["sample_indices"][...], dtype=np.int64)
        umap_attrs = dict(group.attrs.items())

    n_events = int(embedding.shape[0])
    min_cluster_sizes = _parse_ints(args.min_cluster_sizes) or _auto_min_cluster_sizes(n_events)
    min_samples_values = [None] + _parse_ints(args.min_samples) if args.min_samples else _auto_min_samples(min_cluster_sizes)
    candidates = [(size, samples) for size in min_cluster_sizes for samples in min_samples_values]
    if len(candidates) > int(args.max_grid):
        step = max(1, int(np.ceil(len(candidates) / int(args.max_grid))))
        candidates = candidates[::step]

    rows: list[dict[str, object]] = []
    best: tuple[float, int, int | None, np.ndarray, np.ndarray, object, dict[str, float]] | None = None
    for min_cluster_size, min_samples in candidates:
        print(f"fit min_cluster_size={min_cluster_size}, min_samples={min_samples}", flush=True)
        labels, probabilities, clusterer = _fit_hdbscan(embedding, min_cluster_size, min_samples)
        stats = _cluster_stats(labels, probabilities)
        score = _score_candidate(stats, int(args.target_min_clusters), int(args.target_max_clusters))
        row = {
            "min_cluster_size": int(min_cluster_size),
            "min_samples": "" if min_samples is None else int(min_samples),
            "score": float(score),
            **stats,
        }
        rows.append(row)
        print(
            "  clusters={n_clusters:.0f}, noise={noise_fraction:.3f}, "
            "largest={largest_cluster_fraction:.3f}, prob={mean_probability:.3f}, score={score:.3f}".format(
                **row
            ),
            flush=True,
        )
        if best is None or score > best[0]:
            best = (score, int(min_cluster_size), min_samples, labels, probabilities, clusterer, stats)

    if best is None:
        raise RuntimeError("No HDBSCAN candidates were fitted.")

    score, min_cluster_size, min_samples, labels, probabilities, clusterer, stats = best
    cluster_rows: list[dict[str, object]] = []
    for label in sorted(int(x) for x in np.unique(labels)):
        mask = labels == label
        cluster_rows.append(
            {
                "cluster": int(label),
                "events": int(np.count_nonzero(mask)),
                "fraction": float(np.count_nonzero(mask) / labels.size),
                "mean_probability": float(np.mean(probabilities[mask])) if np.any(mask) else 0.0,
                "umap1_mean": float(np.mean(embedding[mask, 0])) if np.any(mask) else np.nan,
                "umap2_mean": float(np.mean(embedding[mask, 1])) if np.any(mask) else np.nan,
            }
        )

    _write_csv(out_dir / "hdbscan_parameter_scan.csv", rows)
    _write_csv(out_dir / "hdbscan_cluster_summary.csv", cluster_rows)

    with h5py.File(out_dir / "hdbscan_umap_clusters.h5", "w") as handle:
        handle.attrs["source_cache"] = str(cache_path)
        handle.attrs["embedding_group"] = str(args.embedding_group)
        handle.attrs["selected_min_cluster_size"] = int(min_cluster_size)
        handle.attrs["selected_min_samples"] = -1 if min_samples is None else int(min_samples)
        handle.attrs["selected_score"] = float(score)
        handle.attrs["selected_stats"] = json.dumps(stats)
        for key, value in umap_attrs.items():
            handle.attrs[f"umap_{key}"] = value
        handle.create_dataset("sample_indices", data=sample_indices, compression="lzf")
        handle.create_dataset("embedding", data=embedding, compression="lzf")
        handle.create_dataset("labels", data=np.asarray(labels, dtype=np.int32), compression="lzf")
        handle.create_dataset("probabilities", data=np.asarray(probabilities, dtype=np.float32), compression="lzf")

    _plot_clusters(
        out_dir / "hdbscan_umap_clusters.png",
        embedding,
        labels,
        probabilities,
        title=(
            "HDBSCAN clusters on clean remaining UMAP\n"
            f"min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
            f"clusters={int(stats['n_clusters'])}, noise={stats['noise_fraction']:.2%}"
        ),
        point_size=float(args.point_size),
    )
    print(f"selected_min_cluster_size: {min_cluster_size}")
    print(f"selected_min_samples: {min_samples}")
    print(f"selected_score: {score:.6f}")
    print(f"clusters: {int(stats['n_clusters'])}")
    print(f"noise_fraction: {stats['noise_fraction']:.6f}")
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
