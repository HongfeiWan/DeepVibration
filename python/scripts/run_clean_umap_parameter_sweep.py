#!/usr/bin/env python
"""Run clean-remaining UMAP parameter sweeps and write evolution panels."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analysis.io.paths import find_project_root  # noqa: E402
from analysis.ml.event_matrix import DEFAULT_CACHE_NAME, compute_basic_anomaly_masks, run_umap_cache  # noqa: E402
from analysis.parallel import add_parallel_arguments  # noqa: E402
from run_clean_remaining_umap import (  # noqa: E402
    DEFAULT_EXCLUDE_MASKS,
    build_clean_remaining_mask,
    plot_clean_remaining_umap,
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
    return root / "data" / "cache" / "clean_umap_parameter_sweep"


def _min_dist_values(args: argparse.Namespace) -> list[float]:
    if args.min_dist_spacing == "linear":
        values = np.linspace(float(args.min_dist_start), float(args.min_dist_end), int(args.min_dist_count))
    else:
        values = np.geomspace(float(args.min_dist_start), float(args.min_dist_end), int(args.min_dist_count))
    return [float(x) for x in values]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_montage(path: Path, image_paths: list[Path], titles: list[str], *, ncols: int = 5) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    n_images = len(image_paths)
    ncols = max(1, min(int(ncols), n_images))
    nrows = int(np.ceil(n_images / ncols))
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


def _umap_sample_size(clean_count: int, requested: int) -> int:
    return clean_count if int(requested) <= 0 else min(int(requested), int(clean_count))


def _run_one(
    *,
    cache_path: Path,
    output_dir: Path,
    clean_mask_name: str,
    clean_count: int,
    n_neighbors: int,
    min_dist: float,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, object]]:
    max_events = _umap_sample_size(clean_count, args.umap_max_events)
    png = output_dir / "clean_remaining_umap.png"
    if args.resume and png.exists():
        row = {
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "clean_available": int(clean_count),
            "umap_sample_events": int(max_events),
            "clean_events_in_sample": int(max_events),
            "output_dir": str(output_dir),
            "png": str(png),
            "status": "skipped_existing",
        }
        return png, row
    run_umap_cache(
        cache_path,
        max_events=max_events,
        random_seed=args.random_seed,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        workers=args.workers,
        chunk_size=args.chunk_size,
        priority_group="anomaly_masks",
        priority_masks=(clean_mask_name,),
        priority_max_events_per_mask=None,
    )
    png = plot_clean_remaining_umap(
        cache_path,
        output_dir=output_dir,
        title=(
            "Clean remaining all-feature UMAP\n"
            "removed: fit-failed, inhibit, RT, ACT, min/pedestal 3sigma, over-threshold\n"
            f"n_neighbors={n_neighbors}, min_dist={min_dist:g}"
        ),
        clean_mask_name=clean_mask_name,
        point_size=args.point_size,
    )
    with h5py.File(cache_path, "r") as handle:
        group = handle["umap"]
        row = {
            "n_neighbors": int(group.attrs["n_neighbors"]),
            "min_dist": float(group.attrs["min_dist"]),
            "clean_available": int(clean_count),
            "umap_sample_events": int(group["sample_indices"].shape[0]),
            "clean_events_in_sample": int(group.attrs.get("priority_events_in_sample", 0)),
            "output_dir": str(output_dir),
            "png": str(png),
            "status": "ran",
        }
    return png, row


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove fit-failed, inhibit, RT, ACT, min/pedestal-3sigma, and over-threshold events, "
            "then run clean-remaining all-feature UMAP sweeps for n_neighbors and min_dist."
        )
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--cache-dir", default=None, help="Default: data/cache/event_feature_umap")
    parser.add_argument("--output-dir", default=None, help="Default: data/cache/clean_umap_parameter_sweep")
    parser.add_argument("--umap-max-events", type=int, default=200_000, help="0 means all clean remaining events.")
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--rt-threshold", type=float, default=6000.0)
    parser.add_argument("--max-adc", type=float, default=16382.0)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--skip-masks", action="store_true", help="Do not recompute anomaly masks before sweeping.")
    parser.add_argument("--clean-mask-name", default="clean_remaining_no_fit_failed")
    parser.add_argument("--exclude-masks", nargs="+", default=list(DEFAULT_EXCLUDE_MASKS))
    parser.add_argument("--fixed-min-dist", type=float, default=0.1)
    parser.add_argument("--fixed-neighbors", type=int, default=400)
    parser.add_argument("--neighbor-start", type=int, default=100)
    parser.add_argument("--neighbor-stop", type=int, default=1000)
    parser.add_argument("--neighbor-step", type=int, default=100)
    parser.add_argument("--min-dist-start", type=float, default=0.001)
    parser.add_argument("--min-dist-end", type=float, default=0.1)
    parser.add_argument("--min-dist-count", type=int, default=10)
    parser.add_argument("--min-dist-spacing", choices=("linear", "log"), default="linear")
    parser.add_argument("--only", choices=("all", "neighbors", "min-dist"), default="all")
    parser.add_argument("--montage-cols", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Skip parameter points whose clean_remaining_umap.png exists.")
    add_parallel_arguments(parser, include_chunk_size=True, default_chunk_size=100000)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cache_path = _cache_path(args)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache does not exist: {cache_path}")
    out_root = _output_root(args)
    out_root.mkdir(parents=True, exist_ok=True)

    setup_rows: list[dict[str, object]] = []
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
            setup_rows.append({"metric": key, "value": int(value)})

    clean_stats = build_clean_remaining_mask(
        cache_path,
        clean_mask_name=args.clean_mask_name,
        exclude_masks=tuple(args.exclude_masks),
        chunk_size=args.chunk_size,
    )
    clean_count = int(clean_stats["clean_remaining"])
    if clean_count <= 0:
        raise RuntimeError("No clean remaining events are available after exclusions.")
    for key, value in clean_stats.items():
        print(f"{key}: {int(value)}", flush=True)
        setup_rows.append({"metric": key, "value": int(value)})
    setup_rows.append({"metric": "umap_requested_sample_events", "value": _umap_sample_size(clean_count, args.umap_max_events)})
    _write_csv(out_root / "clean_umap_sweep_setup_summary.csv", setup_rows)

    rows: list[dict[str, object]] = []

    if args.only in {"all", "neighbors"}:
        image_paths: list[Path] = []
        titles: list[str] = []
        neighbors = list(range(int(args.neighbor_start), int(args.neighbor_stop) + 1, int(args.neighbor_step)))
        for n_neighbors in neighbors:
            out_dir = out_root / "neighbors" / f"nn{n_neighbors:04d}_md{_tag_float(args.fixed_min_dist)}"
            print(f"[neighbors] n_neighbors={n_neighbors}, min_dist={args.fixed_min_dist:g}", flush=True)
            png, row = _run_one(
                cache_path=cache_path,
                output_dir=out_dir,
                clean_mask_name=args.clean_mask_name,
                clean_count=clean_count,
                n_neighbors=n_neighbors,
                min_dist=float(args.fixed_min_dist),
                args=args,
            )
            row["sweep"] = "neighbors"
            image_paths.append(png)
            titles.append(f"n_neighbors={n_neighbors}")
            rows.append(row)
        _write_montage(out_root / "neighbors_sweep_clean_grid.png", image_paths, titles, ncols=args.montage_cols)

    if args.only in {"all", "min-dist"}:
        image_paths = []
        titles = []
        for min_dist in _min_dist_values(args):
            out_dir = out_root / "min_dist" / f"nn{args.fixed_neighbors:04d}_md{_tag_float(min_dist)}"
            print(f"[min-dist] n_neighbors={args.fixed_neighbors}, min_dist={min_dist:g}", flush=True)
            png, row = _run_one(
                cache_path=cache_path,
                output_dir=out_dir,
                clean_mask_name=args.clean_mask_name,
                clean_count=clean_count,
                n_neighbors=int(args.fixed_neighbors),
                min_dist=float(min_dist),
                args=args,
            )
            row["sweep"] = "min_dist"
            image_paths.append(png)
            titles.append(f"min_dist={min_dist:.4g}")
            rows.append(row)
        _write_montage(out_root / "min_dist_sweep_clean_grid.png", image_paths, titles, ncols=args.montage_cols)

    _write_csv(out_root / "clean_umap_parameter_sweep_summary.csv", rows)
    print(f"Wrote {out_root}", flush=True)


if __name__ == "__main__":
    main()
