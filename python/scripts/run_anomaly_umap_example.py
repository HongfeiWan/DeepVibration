#!/usr/bin/env python
"""Highlight known physics-anomaly classes on the full-feature UMAP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.io.paths import find_project_root  # noqa: E402
from analysis.ml.event_matrix import (  # noqa: E402
    DEFAULT_CACHE_NAME,
    compute_basic_anomaly_masks,
    plot_anomaly_umap,
    run_umap_cache,
)
from analysis.parallel import add_parallel_arguments  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a compact all-feature UMAP and color only physics-motivated "
            "anomaly classes: inhibit, RT, pedestal 3sigma, over-threshold, ACT, and min 3sigma. "
            "By default the UMAP sample is anomaly-prioritized so rare basic-cut "
            "failures are not lost in a random draw."
        )
    )
    parser.add_argument("--stage", choices=["all", "umap", "masks", "plot"], default="all")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--cache-dir", default=None, help="Default: data/cache/event_feature_umap")
    parser.add_argument("--umap-max-events", type=int, default=200_000, help="0 means fit UMAP on all events.")
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--umap-neighbors", type=int, default=150)
    parser.add_argument("--umap-min-dist", type=float, default=0.005)
    parser.add_argument("--rt-threshold", type=float, default=6000.0)
    parser.add_argument("--max-adc", type=float, default=16382.0)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--output-dir", default=None, help="Default: data/cache/event_feature_umap/anomaly_umap_*")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument(
        "--stratify-anomalies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include anomaly-mask events in the UMAP fit/sample before filling with random background.",
    )
    parser.add_argument(
        "--max-events-per-anomaly",
        type=int,
        default=20000,
        help="Cap events injected from each anomaly mask before filling random background; <=0 keeps all.",
    )
    add_parallel_arguments(parser, include_chunk_size=True, default_chunk_size=100000)
    return parser.parse_args(argv)


def _cache_path(args: argparse.Namespace) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir) / DEFAULT_CACHE_NAME
    root = find_project_root(args.project_root)
    return root / "data" / "cache" / "event_feature_umap" / DEFAULT_CACHE_NAME


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    root = find_project_root(args.project_root)
    md = str(args.umap_min_dist).replace(".", "p")
    tag = f"nn{args.umap_neighbors}_md{md}"
    return root / "data" / "cache" / "event_feature_umap" / f"anomaly_umap_{tag}"


def main(argv=None) -> None:
    args = parse_args(argv)
    cache_path = _cache_path(args)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache does not exist: {cache_path}")

    if args.stage in {"all", "masks"}:
        stats = compute_basic_anomaly_masks(
            cache_path,
            rt_threshold=args.rt_threshold,
            max_adc=args.max_adc,
            n_sigma=args.sigma,
            chunk_size=args.chunk_size,
        )
        for key, value in stats.items():
            print(f"{key}: {int(value)}")

    if args.stage in {"all", "umap"}:
        max_per_anomaly = None if args.max_events_per_anomaly <= 0 else args.max_events_per_anomaly
        run_umap_cache(
            cache_path,
            max_events=args.umap_max_events,
            random_seed=args.random_seed,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            workers=args.workers,
            chunk_size=args.chunk_size,
            priority_group="anomaly_masks" if args.stratify_anomalies else None,
            priority_masks=(
                "inhibit",
                "random_trigger",
                "pedestal_3sigma_outlier",
                "over_threshold",
                "act",
                "min_3sigma_outlier",
            ) if args.stratify_anomalies else None,
            priority_max_events_per_mask=max_per_anomaly,
        )
        print(f"UMAP embedding cached in {cache_path}")

    if args.stage in {"all", "plot"}:
        out = _output_dir(args)
        paths = plot_anomaly_umap(
            cache_path,
            output_dir=out,
            point_size=args.point_size,
            title=(
                f"All-feature UMAP anomaly example "
                f"(n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})"
            ),
        )
        for path in paths:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
