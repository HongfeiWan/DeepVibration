#!/usr/bin/env python
"""Run per-channel event-feature UMAP diagnostics from the shared cache."""

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
    plot_umap_masks,
    run_feature_subset_umap_cache,
)
from analysis.parallel import add_parallel_arguments  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run separate Mahalanobis UMAP embeddings for CH0/CH1/CH2/CH3 "
            "feature subsets while coloring with the shared physics masks."
        )
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--cache-dir", default=None, help="Default: data/cache/event_feature_umap")
    parser.add_argument("--channels", nargs="+", default=["CH0", "CH1", "CH2", "CH3"])
    parser.add_argument("--umap-max-events", type=int, default=200_000, help="0 means fit UMAP on all events.")
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--umap-neighbors", type=int, default=80)
    parser.add_argument("--umap-min-dist", type=float, default=0.02)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--output-dir", default=None, help="Default: data/cache/event_feature_umap/channel_umaps")
    add_parallel_arguments(parser, include_chunk_size=True, default_chunk_size=100000)
    return parser.parse_args(argv)


def _cache_path(args: argparse.Namespace) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir) / DEFAULT_CACHE_NAME
    root = find_project_root(args.project_root)
    return root / "data" / "cache" / "event_feature_umap" / DEFAULT_CACHE_NAME


def _output_root(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    root = find_project_root(args.project_root)
    tag = f"nn{args.umap_neighbors}_md{str(args.umap_min_dist).replace('.', 'p')}"
    return root / "data" / "cache" / "event_feature_umap" / f"channel_umaps_{tag}"


def main(argv=None) -> None:
    args = parse_args(argv)
    cache_path = _cache_path(args)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache does not exist: {cache_path}")

    output_root = _output_root(args)
    output_root.mkdir(parents=True, exist_ok=True)
    for channel in args.channels:
        label = channel.upper()
        prefix = f"{label.lower()}_"
        group_name = f"umap_by_channel/{label.lower()}"
        run_feature_subset_umap_cache(
            cache_path,
            feature_prefixes=[prefix],
            group_name=group_name,
            max_events=args.umap_max_events,
            random_seed=args.random_seed,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )
        paths = plot_umap_masks(
            cache_path,
            output_dir=output_root / label.lower(),
            point_size=args.point_size,
            embedding_group=group_name,
            title=f"{label} feature UMAP colored by physics masks",
            filename_prefix=f"{label.lower()}_umap",
        )
        for path in paths:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
