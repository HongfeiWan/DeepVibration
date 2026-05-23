#!/usr/bin/env python
"""Build event feature cache and Mahalanobis-UMAP mask diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.ml.event_matrix import (  # noqa: E402
    DEFAULT_CACHE_NAME,
    build_feature_cache,
    fit_ledoitwolf_metric,
    plot_umap_masks,
    run_umap_cache,
)
from analysis.io.paths import find_project_root  # noqa: E402
from analysis.parallel import add_parallel_arguments  # noqa: E402


def _parse_range(text: str) -> tuple[float, float]:
    normalized = text.replace(",", ":")
    parts = normalized.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be START:STOP or START,STOP")
    start, stop = float(parts[0]), float(parts[1])
    if stop <= start:
        raise argparse.ArgumentTypeError("range STOP must be greater than START")
    return start, stop


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache all paired CH*_parameters event features, fit LedoitWolf covariance, "
            "run Mahalanobis UMAP, and plot mask-colored diagnostics."
        )
    )
    parser.add_argument("--stage", choices=["all", "cache", "fit", "umap", "plot"], default="all")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--cache-dir", default=None, help="Default: data/cache/event_feature_umap")
    parser.add_argument("--channels", nargs="+", default=None, help="Default: all discovered CH*_parameters folders.")
    parser.add_argument("--max-runs", type=int, default=None, help="Debug limit for the number of paired runs.")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    parser.add_argument("--cov-fit-max-events", type=int, default=1_000_000, help="0 means fit covariance on all events.")
    parser.add_argument("--umap-max-events", type=int, default=200_000, help="0 means fit UMAP on all events.")
    parser.add_argument("--transform-all", action="store_true", help="Also transform every cached event in chunks.")
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--pn-fit-ch0-min", type=float, default=5000.0)
    parser.add_argument("--pn-fit-ch0-max", type=float, default=16000.0)
    parser.add_argument("--pn-sigma", type=float, default=0.8)
    parser.add_argument(
        "--time-keep-range",
        action="append",
        type=_parse_range,
        default=[],
        help="Keep only events inside START:STOP in cached event_time units. Repeatable.",
    )
    parser.add_argument(
        "--bad-time-interval",
        action="append",
        type=_parse_range,
        default=[],
        help="Reject events inside START:STOP in cached event_time units. Repeatable.",
    )
    parser.add_argument(
        "--time-unit",
        choices=["raw", "unix"],
        default="raw",
        help="raw keeps HDF5 time_data; unix subtracts the DAQ epoch offset before time cuts.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for UMAP PNG outputs. Default: cache dir.")
    parser.add_argument("--point-size", type=float, default=3.0)
    add_parallel_arguments(parser, include_chunk_size=True, default_chunk_size=10000)
    return parser.parse_args(argv)


def _cache_path(args: argparse.Namespace) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir) / DEFAULT_CACHE_NAME
    root = find_project_root(args.project_root)
    return root / "data" / "cache" / "event_feature_umap" / DEFAULT_CACHE_NAME


def main(argv=None) -> None:
    args = parse_args(argv)
    cache_path = _cache_path(args)

    if args.stage in {"all", "cache"}:
        compression = None if args.compression == "none" else args.compression
        cache_path = build_feature_cache(
            project_root=args.project_root,
            cache_dir=args.cache_dir,
            channels=args.channels,
            workers=args.workers,
            chunk_size=args.chunk_size,
            compression=compression,
            rebuild=args.rebuild_cache,
            max_runs=args.max_runs,
            pn_fit_ch0_min=args.pn_fit_ch0_min,
            pn_fit_ch0_max=args.pn_fit_ch0_max,
            pn_sigma=args.pn_sigma,
            time_keep_ranges=args.time_keep_range,
            bad_time_intervals=args.bad_time_interval,
            time_unit=args.time_unit,
        )
        print(f"feature cache: {cache_path}")

    if args.stage in {"all", "fit"}:
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache does not exist: {cache_path}")
        fit_ledoitwolf_metric(
            cache_path,
            max_fit_events=args.cov_fit_max_events,
            random_seed=args.random_seed,
            chunk_size=args.chunk_size,
            workers=args.workers,
        )
        print(f"LedoitWolf Mahalanobis metric cached in {cache_path}")

    if args.stage in {"all", "umap"}:
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache does not exist: {cache_path}")
        run_umap_cache(
            cache_path,
            max_events=args.umap_max_events,
            random_seed=args.random_seed,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            workers=args.workers,
            transform_all=args.transform_all,
            chunk_size=args.chunk_size,
        )
        print(f"UMAP embedding cached in {cache_path}")

    if args.stage in {"all", "plot"}:
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache does not exist: {cache_path}")
        paths = plot_umap_masks(cache_path, output_dir=args.output_dir, point_size=args.point_size)
        for path in paths:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
