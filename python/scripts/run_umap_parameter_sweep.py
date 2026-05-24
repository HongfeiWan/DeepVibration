#!/usr/bin/env python
"""Run anomaly-UMAP parameter sweeps and write evolution panels."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

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


PRIORITY_MASKS = (
    "inhibit",
    "random_trigger",
    "pedestal_3sigma_outlier",
    "over_threshold",
    "act",
    "min_3sigma_outlier",
)


def _tag_float(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace(".", "p").replace("-", "m")


def _cache_path(args: argparse.Namespace) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir) / DEFAULT_CACHE_NAME
    root = find_project_root(args.project_root)
    return root / "data" / "cache" / "event_feature_umap" / DEFAULT_CACHE_NAME


def _output_root(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    root = find_project_root(args.project_root)
    return root / "data" / "cache" / "umap_parameter_sweep"


def _min_dist_values(args: argparse.Namespace) -> list[float]:
    if args.min_dist_spacing == "linear":
        values = np.linspace(float(args.min_dist_start), float(args.min_dist_end), int(args.min_dist_count))
    else:
        values = np.geomspace(float(args.min_dist_start), float(args.min_dist_end), int(args.min_dist_count))
    return [float(x) for x in values]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_montage(path: Path, image_paths: list[Path], titles: list[str], *, ncols: int = 5) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(image_paths)
    ncols = max(1, min(int(ncols), n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for ax, image_path, title in zip(axes_arr.ravel(), image_paths, titles):
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_one(
    *,
    cache_path: Path,
    output_dir: Path,
    n_neighbors: int,
    min_dist: float,
    args: argparse.Namespace,
) -> Path:
    max_per_anomaly = None if args.max_events_per_anomaly <= 0 else int(args.max_events_per_anomaly)
    run_umap_cache(
        cache_path,
        max_events=args.umap_max_events,
        random_seed=args.random_seed,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        workers=args.workers,
        chunk_size=args.chunk_size,
        priority_group="anomaly_masks",
        priority_masks=PRIORITY_MASKS,
        priority_max_events_per_mask=max_per_anomaly,
    )
    paths = plot_anomaly_umap(
        cache_path,
        output_dir=output_dir,
        point_size=args.point_size,
        title=f"All-feature UMAP anomaly example (n_neighbors={n_neighbors}, min_dist={min_dist:g})",
    )
    highlighted = output_dir / "anomaly_umap_highlighted.png"
    if highlighted not in paths and not highlighted.exists():
        raise FileNotFoundError(f"Expected highlighted plot was not written: {highlighted}")
    return highlighted


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two anomaly-UMAP evolution sweeps: n_neighbors and min_dist. "
            "Each UMAP run is sequential; each run may use all workers internally."
        )
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--cache-dir", default=None, help="Default: data/cache/event_feature_umap")
    parser.add_argument("--output-dir", default=None, help="Default: data/cache/umap_parameter_sweep")
    parser.add_argument("--umap-max-events", type=int, default=200_000, help="0 means fit UMAP on all events.")
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--max-events-per-anomaly", type=int, default=20_000)
    parser.add_argument("--fixed-min-dist", type=float, default=0.001)
    parser.add_argument("--fixed-neighbors", type=int, default=400)
    parser.add_argument("--neighbor-start", type=int, default=100)
    parser.add_argument("--neighbor-stop", type=int, default=1000)
    parser.add_argument("--neighbor-step", type=int, default=100)
    parser.add_argument("--min-dist-start", type=float, default=0.001)
    parser.add_argument("--min-dist-end", type=float, default=0.1)
    parser.add_argument("--min-dist-count", type=int, default=10)
    parser.add_argument("--min-dist-spacing", choices=("log", "linear"), default="log")
    parser.add_argument("--skip-masks", action="store_true", help="Do not recompute anomaly masks before the sweep.")
    parser.add_argument("--only", choices=("all", "neighbors", "min-dist"), default="all")
    add_parallel_arguments(parser, include_chunk_size=True, default_chunk_size=100000)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cache_path = _cache_path(args)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache does not exist: {cache_path}")
    out_root = _output_root(args)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_masks:
        stats = compute_basic_anomaly_masks(cache_path, chunk_size=args.chunk_size)
        for key, value in stats.items():
            print(f"{key}: {int(value)}", flush=True)

    rows: list[dict[str, object]] = []

    if args.only in {"all", "neighbors"}:
        image_paths: list[Path] = []
        titles: list[str] = []
        neighbors = list(range(int(args.neighbor_start), int(args.neighbor_stop) + 1, int(args.neighbor_step)))
        for n_neighbors in neighbors:
            out_dir = out_root / "neighbors" / f"nn{n_neighbors:04d}_md{_tag_float(args.fixed_min_dist)}"
            print(f"[neighbors] n_neighbors={n_neighbors}, min_dist={args.fixed_min_dist:g}", flush=True)
            highlighted = _run_one(
                cache_path=cache_path,
                output_dir=out_dir,
                n_neighbors=n_neighbors,
                min_dist=float(args.fixed_min_dist),
                args=args,
            )
            image_paths.append(highlighted)
            titles.append(f"n_neighbors={n_neighbors}")
            rows.append(
                {
                    "sweep": "neighbors",
                    "n_neighbors": n_neighbors,
                    "min_dist": float(args.fixed_min_dist),
                    "output_dir": str(out_dir),
                    "highlighted_png": str(highlighted),
                }
            )
        _write_montage(out_root / "neighbors_sweep_highlighted_grid.png", image_paths, titles)

    if args.only in {"all", "min-dist"}:
        image_paths = []
        titles = []
        min_dists = _min_dist_values(args)
        for min_dist in min_dists:
            out_dir = out_root / "min_dist" / f"nn{args.fixed_neighbors:04d}_md{_tag_float(min_dist)}"
            print(f"[min-dist] n_neighbors={args.fixed_neighbors}, min_dist={min_dist:g}", flush=True)
            highlighted = _run_one(
                cache_path=cache_path,
                output_dir=out_dir,
                n_neighbors=int(args.fixed_neighbors),
                min_dist=float(min_dist),
                args=args,
            )
            image_paths.append(highlighted)
            titles.append(f"min_dist={min_dist:.4g}")
            rows.append(
                {
                    "sweep": "min_dist",
                    "n_neighbors": int(args.fixed_neighbors),
                    "min_dist": float(min_dist),
                    "output_dir": str(out_dir),
                    "highlighted_png": str(highlighted),
                }
            )
        _write_montage(out_root / "min_dist_sweep_highlighted_grid.png", image_paths, titles)

    if rows:
        _write_csv(out_root / "umap_parameter_sweep_summary.csv", rows)
    print(f"Wrote {out_root}", flush=True)


if __name__ == "__main__":
    main()
