"""Machine-learning helpers for parameterized event studies."""

from .clustering import ledoit_wolf_covariance, run_gmm, run_hdbscan
from .embedding import EmbeddingResult, feature_matrix, run_pca, run_umap, standardize_features
from .event_matrix import (
    build_feature_cache,
    compute_basic_anomaly_masks,
    discover_parameter_channels,
    fit_ledoitwolf_metric,
    plot_anomaly_umap,
    plot_umap_masks,
    run_feature_subset_umap_cache,
    run_umap_cache,
)

__all__ = [
    "EmbeddingResult",
    "build_feature_cache",
    "compute_basic_anomaly_masks",
    "discover_parameter_channels",
    "feature_matrix",
    "fit_ledoitwolf_metric",
    "ledoit_wolf_covariance",
    "plot_anomaly_umap",
    "plot_umap_masks",
    "run_feature_subset_umap_cache",
    "run_gmm",
    "run_hdbscan",
    "run_pca",
    "run_umap_cache",
    "run_umap",
    "standardize_features",
]
