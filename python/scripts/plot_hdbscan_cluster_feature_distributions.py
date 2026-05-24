#!/usr/bin/env python
"""Plot feature distributions for HDBSCAN clusters on a cached UMAP sample."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import h5py
import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.io.paths import find_project_root  # noqa: E402


def _default_cluster_path(root: Path) -> Path:
    return (
        root
        / "data"
        / "cache"
        / "hdbscan_clean_remaining_umap_exclude_fit_failed_6clusters"
        / "hdbscan_umap_clusters.h5"
    )


def _default_feature_cache(root: Path) -> Path:
    return root / "data" / "cache" / "event_feature_umap" / "event_feature_cache.h5"


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _cluster_name(label: int) -> str:
    return "noise" if int(label) < 0 else f"C{int(label)}"


def _chunk_ranges(n_items: int, chunk_size: int):
    for start in range(0, int(n_items), int(chunk_size)):
        yield start, min(start + int(chunk_size), int(n_items))


def _read_feature_rows(
    dataset: h5py.Dataset,
    indices: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    order = np.argsort(idx)
    sorted_idx = idx[order]
    out = np.empty((idx.size, int(dataset.shape[1])), dtype=np.float32)
    n_events = int(dataset.shape[0])
    for start, stop in _chunk_ranges(n_events, int(chunk_size)):
        left = int(np.searchsorted(sorted_idx, start, side="left"))
        right = int(np.searchsorted(sorted_idx, stop, side="left"))
        if right <= left:
            continue
        block = np.asarray(dataset[start:stop, :], dtype=np.float32)
        out[order[left:right], :] = block[sorted_idx[left:right] - start, :]
    return out


def _robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q25 = np.nanpercentile(values, 25.0, axis=0)
    q50 = np.nanpercentile(values, 50.0, axis=0)
    q75 = np.nanpercentile(values, 75.0, axis=0)
    scale = q75 - q25
    fallback = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    q50 = np.where(np.isfinite(q50), q50, 0.0)
    return q50.astype(np.float64), scale.astype(np.float64)


def _separation_scores(
    features: np.ndarray,
    labels: np.ndarray,
    cluster_labels: np.ndarray,
    global_center: np.ndarray,
    global_scale: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray]:
    rows: list[dict[str, object]] = []
    z_medians = np.full((cluster_labels.size, features.shape[1]), np.nan, dtype=np.float64)
    for j in range(features.shape[1]):
        x = np.asarray(features[:, j], dtype=np.float64)
        finite = np.isfinite(x)
        if np.count_nonzero(finite) < 2:
            rows.append(
                {
                    "feature_index": j,
                    "eta2_with_noise": 0.0,
                    "eta2_no_noise": 0.0,
                    "robust_median_range": 0.0,
                    "finite_count": int(np.count_nonzero(finite)),
                }
            )
            continue

        medians: list[float] = []
        for row_idx, label in enumerate(cluster_labels):
            mask = finite & (labels == label)
            if np.any(mask):
                median = float(np.nanmedian(x[mask]))
                z_medians[row_idx, j] = (median - global_center[j]) / global_scale[j]
                medians.append(z_medians[row_idx, j])

        def eta2(use_noise: bool) -> float:
            mask = finite if use_noise else finite & (labels >= 0)
            if np.count_nonzero(mask) < 2:
                return 0.0
            total_mean = float(np.nanmean(x[mask]))
            total_ss = float(np.nansum((x[mask] - total_mean) ** 2))
            if total_ss <= 0.0 or not np.isfinite(total_ss):
                return 0.0
            between_ss = 0.0
            for label in cluster_labels:
                if (not use_noise) and int(label) < 0:
                    continue
                group = mask & (labels == label)
                n_group = int(np.count_nonzero(group))
                if n_group == 0:
                    continue
                group_mean = float(np.nanmean(x[group]))
                between_ss += n_group * (group_mean - total_mean) ** 2
            value = between_ss / total_ss
            return float(value) if np.isfinite(value) else 0.0

        rows.append(
            {
                "feature_index": j,
                "eta2_with_noise": eta2(True),
                "eta2_no_noise": eta2(False),
                "robust_median_range": float(np.nanmax(medians) - np.nanmin(medians)) if medians else 0.0,
                "finite_count": int(np.count_nonzero(finite)),
            }
        )
    return rows, z_medians


def _cluster_feature_summary(
    features: np.ndarray,
    labels: np.ndarray,
    cluster_labels: np.ndarray,
    feature_names: list[str],
    z_medians: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_idx, label in enumerate(cluster_labels):
        cluster_mask = labels == label
        for feature_idx, feature_name in enumerate(feature_names):
            x = np.asarray(features[cluster_mask, feature_idx], dtype=np.float64)
            x = x[np.isfinite(x)]
            if x.size == 0:
                stats = {
                    "events": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "q05": np.nan,
                    "q25": np.nan,
                    "median": np.nan,
                    "q75": np.nan,
                    "q95": np.nan,
                    "robust_z_median": np.nan,
                }
            else:
                stats = {
                    "events": int(x.size),
                    "mean": float(np.mean(x)),
                    "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
                    "q05": float(np.percentile(x, 5.0)),
                    "q25": float(np.percentile(x, 25.0)),
                    "median": float(np.percentile(x, 50.0)),
                    "q75": float(np.percentile(x, 75.0)),
                    "q95": float(np.percentile(x, 95.0)),
                    "robust_z_median": float(z_medians[row_idx, feature_idx]),
                }
            rows.append(
                {
                    "cluster": int(label),
                    "cluster_name": _cluster_name(int(label)),
                    "feature_index": int(feature_idx),
                    "feature": feature_name,
                    **stats,
                }
            )
    return rows


def _cluster_top_feature_rows(
    summary_rows: list[dict[str, object]],
    *,
    per_cluster: int,
) -> list[dict[str, object]]:
    by_cluster: dict[str, list[dict[str, object]]] = {}
    for row in summary_rows:
        by_cluster.setdefault(str(row["cluster_name"]), []).append(row)
    out: list[dict[str, object]] = []
    for cluster_name, rows in sorted(by_cluster.items()):
        ordered = sorted(
            rows,
            key=lambda item: abs(float(item["robust_z_median"]))
            if np.isfinite(float(item["robust_z_median"]))
            else -1.0,
            reverse=True,
        )
        for rank, row in enumerate(ordered[:per_cluster], start=1):
            out.append(
                {
                    "cluster": row["cluster"],
                    "cluster_name": cluster_name,
                    "rank": rank,
                    "feature_index": row["feature_index"],
                    "feature": row["feature"],
                    "robust_z_median": row["robust_z_median"],
                    "median": row["median"],
                    "q25": row["q25"],
                    "q75": row["q75"],
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_heatmap(
    path: Path,
    *,
    z_medians: np.ndarray,
    feature_names: list[str],
    cluster_labels: np.ndarray,
    top_indices: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    matrix = z_medians[:, top_indices].T
    vmax = float(np.nanpercentile(np.abs(matrix), 98.0))
    vmax = max(vmax, 1.0)
    fig_height = max(7.0, 0.32 * top_indices.size)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(cluster_labels.size))
    ax.set_xticklabels([_cluster_name(int(label)) for label in cluster_labels], rotation=0)
    ax.set_yticks(np.arange(top_indices.size))
    ax.set_yticklabels([feature_names[int(i)] for i in top_indices], fontsize=8)
    ax.set_title("Cluster median feature shifts (robust z-score)")
    fig.colorbar(im, ax=ax, label="cluster median vs global median / IQR")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_boxplot_pages(
    output_dir: Path,
    *,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    cluster_labels: np.ndarray,
    top_indices: np.ndarray,
    features_per_page: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    cluster_names = [_cluster_name(int(label)) for label in cluster_labels]
    for page, start in enumerate(range(0, top_indices.size, int(features_per_page)), start=1):
        indices = top_indices[start : start + int(features_per_page)]
        n = int(indices.size)
        ncols = 3
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(4.2, 3.0 * nrows)))
        axes = np.asarray(axes).reshape(-1)
        for ax_idx, ax in enumerate(axes):
            if ax_idx >= n:
                ax.axis("off")
                continue
            feature_idx = int(indices[ax_idx])
            data = []
            for label in cluster_labels:
                x = np.asarray(features[labels == label, feature_idx], dtype=np.float64)
                data.append(x[np.isfinite(x)])
            ax.boxplot(data, tick_labels=cluster_names, showfliers=False, widths=0.68)
            ax.set_title(feature_names[feature_idx], fontsize=9)
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
        fig.suptitle("Top separating feature distributions by HDBSCAN cluster", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = output_dir / f"feature_boxplots_top_page_{page:02d}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_histograms(
    path: Path,
    *,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    cluster_labels: np.ndarray,
    top_indices: np.ndarray,
    max_features: int,
    max_points_per_cluster: int,
    random_seed: int,
) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(int(random_seed))
    selected_features = top_indices[: int(max_features)]
    n = int(selected_features.size)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(4.5, 3.2 * nrows)))
    axes = np.asarray(axes).reshape(-1)
    for ax_idx, ax in enumerate(axes):
        if ax_idx >= n:
            ax.axis("off")
            continue
        feature_idx = int(selected_features[ax_idx])
        x_all = np.asarray(features[:, feature_idx], dtype=np.float64)
        finite_all = x_all[np.isfinite(x_all)]
        if finite_all.size == 0:
            ax.axis("off")
            continue
        lo, hi = np.percentile(finite_all, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(finite_all)), float(np.nanmax(finite_all))
        bins = np.linspace(lo, hi, 60)
        for label_idx, label in enumerate(cluster_labels):
            x = x_all[(labels == label) & np.isfinite(x_all)]
            if x.size == 0:
                continue
            if x.size > int(max_points_per_cluster):
                x = rng.choice(x, size=int(max_points_per_cluster), replace=False)
            ax.hist(
                x,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.2,
                color=colors(label_idx % 10),
                label=_cluster_name(int(label)),
            )
        ax.set_title(feature_names[feature_idx], fontsize=9)
        ax.tick_params(labelsize=8)
    handles, labels_text = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="upper right", fontsize=8)
    fig.suptitle("Density histograms for top separating features", fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.94, 0.97))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_umap_feature_panels(
    path: Path,
    *,
    embedding: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    top_indices: np.ndarray,
    max_features: int,
    max_scatter_points: int,
    random_seed: int,
) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(int(random_seed))
    n_points = int(embedding.shape[0])
    if n_points > int(max_scatter_points):
        points = np.sort(rng.choice(n_points, size=int(max_scatter_points), replace=False))
    else:
        points = np.arange(n_points)
    selected_features = top_indices[: int(max_features)]
    n = int(selected_features.size)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(4.5, 4.0 * nrows)))
    axes = np.asarray(axes).reshape(-1)
    for ax_idx, ax in enumerate(axes):
        if ax_idx >= n:
            ax.axis("off")
            continue
        feature_idx = int(selected_features[ax_idx])
        values = np.asarray(features[points, feature_idx], dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            ax.axis("off")
            continue
        lo, hi = np.nanpercentile(values[finite], [2.0, 98.0])
        clipped = np.clip(values, lo, hi)
        sc = ax.scatter(
            embedding[points, 0],
            embedding[points, 1],
            c=clipped,
            s=1.2,
            alpha=0.65,
            linewidths=0,
            cmap="viridis",
        )
        ax.set_title(feature_names[feature_idx], fontsize=9)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("UMAP colored by top separating parameters", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_top_pair_scatter(
    path: Path,
    *,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    cluster_labels: np.ndarray,
    top_indices: np.ndarray,
    max_points: int,
    random_seed: int,
) -> None:
    import matplotlib.pyplot as plt

    if top_indices.size < 2:
        return
    rng = np.random.default_rng(int(random_seed))
    n_points = int(features.shape[0])
    if n_points > int(max_points):
        points = np.sort(rng.choice(n_points, size=int(max_points), replace=False))
    else:
        points = np.arange(n_points)
    x_idx = int(top_indices[0])
    y_idx = int(top_indices[1])
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    for label_idx, label in enumerate(cluster_labels):
        mask = labels[points] == label
        x = features[points[mask], x_idx]
        y = features[points[mask], y_idx]
        finite = np.isfinite(x) & np.isfinite(y)
        ax.scatter(
            x[finite],
            y[finite],
            s=2.0,
            alpha=0.45,
            linewidths=0,
            color=colors(label_idx % 10),
            label=_cluster_name(int(label)),
        )
    ax.set_xlabel(feature_names[x_idx])
    ax.set_ylabel(feature_names[y_idx])
    ax.set_title("Top two separating parameters")
    ax.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot feature distributions for HDBSCAN clusters.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--cluster-h5", default=None)
    parser.add_argument("--feature-cache", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--box-features-per-page", type=int, default=12)
    parser.add_argument("--hist-top-n", type=int, default=12)
    parser.add_argument("--umap-feature-top-n", type=int, default=9)
    parser.add_argument("--cluster-top-n", type=int, default=10)
    parser.add_argument("--max-points-per-cluster", type=int, default=30000)
    parser.add_argument("--max-scatter-points", type=int, default=200000)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    root = find_project_root(args.project_root)
    cluster_h5 = Path(args.cluster_h5) if args.cluster_h5 else _default_cluster_path(root)
    feature_cache = Path(args.feature_cache) if args.feature_cache else _default_feature_cache(root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "data" / "cache" / "hdbscan_cluster_feature_distributions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(cluster_h5, "r") as handle:
        labels = np.asarray(handle["labels"][...], dtype=np.int64)
        embedding = np.asarray(handle["embedding"][...], dtype=np.float32)
        sample_indices = np.asarray(handle["sample_indices"][...], dtype=np.int64)

    with h5py.File(feature_cache, "r") as handle:
        feature_names = [_decode(x) for x in handle["feature_names"][...]]
        features = _read_feature_rows(handle["features"], sample_indices, chunk_size=int(args.chunk_size))

    cluster_labels = np.asarray(sorted(int(x) for x in np.unique(labels)), dtype=np.int64)
    center, scale = _robust_center_scale(features)
    score_rows, z_medians = _separation_scores(features, labels, cluster_labels, center, scale)
    for row in score_rows:
        idx = int(row["feature_index"])
        row["feature"] = feature_names[idx]
    score_rows = sorted(score_rows, key=lambda item: float(item["eta2_no_noise"]), reverse=True)
    _write_csv(output_dir / "feature_separation_scores.csv", score_rows)

    summary_rows = _cluster_feature_summary(features, labels, cluster_labels, feature_names, z_medians)
    _write_csv(output_dir / "cluster_feature_summary.csv", summary_rows)
    _write_csv(
        output_dir / "cluster_top_features.csv",
        _cluster_top_feature_rows(summary_rows, per_cluster=int(args.cluster_top_n)),
    )

    top_indices = np.asarray([int(row["feature_index"]) for row in score_rows[: int(args.top_n)]], dtype=np.int64)
    _plot_heatmap(
        output_dir / "cluster_median_heatmap_top_features.png",
        z_medians=z_medians,
        feature_names=feature_names,
        cluster_labels=cluster_labels,
        top_indices=top_indices,
    )
    _plot_boxplot_pages(
        output_dir,
        features=features,
        labels=labels,
        feature_names=feature_names,
        cluster_labels=cluster_labels,
        top_indices=top_indices,
        features_per_page=int(args.box_features_per_page),
    )
    _plot_histograms(
        output_dir / "feature_histograms_top.png",
        features=features,
        labels=labels,
        feature_names=feature_names,
        cluster_labels=cluster_labels,
        top_indices=top_indices,
        max_features=int(args.hist_top_n),
        max_points_per_cluster=int(args.max_points_per_cluster),
        random_seed=int(args.random_seed),
    )
    _plot_umap_feature_panels(
        output_dir / "umap_top_feature_gradients.png",
        embedding=embedding,
        features=features,
        feature_names=feature_names,
        top_indices=top_indices,
        max_features=int(args.umap_feature_top_n),
        max_scatter_points=int(args.max_scatter_points),
        random_seed=int(args.random_seed),
    )
    _plot_top_pair_scatter(
        output_dir / "top_two_feature_scatter.png",
        features=features,
        labels=labels,
        feature_names=feature_names,
        cluster_labels=cluster_labels,
        top_indices=top_indices,
        max_points=int(args.max_scatter_points),
        random_seed=int(args.random_seed),
    )
    print(f"Wrote {output_dir}", flush=True)
    print("Top features:", ", ".join(feature_names[int(i)] for i in top_indices[:10]), flush=True)


if __name__ == "__main__":
    main()
