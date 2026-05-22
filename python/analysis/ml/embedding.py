"""Reusable dimensionality-reduction helpers for parameter studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from analysis.parallel import ParallelConfig


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: np.ndarray
    labels: np.ndarray | None = None


def standardize_features(features: np.ndarray) -> np.ndarray:
    try:
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise ImportError("Install the 'ml' extra to standardize ML features") from exc

    return StandardScaler().fit_transform(np.asarray(features, dtype=np.float64))


def run_pca(features: np.ndarray, *, n_components: int = 2, standardize: bool = True) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:
        raise ImportError("Install the 'ml' extra to run PCA") from exc

    x = standardize_features(features) if standardize else np.asarray(features, dtype=np.float64)
    return PCA(n_components=n_components).fit_transform(x)


def run_umap(
    features: np.ndarray,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = None,
    standardize: bool = True,
    config: ParallelConfig | None = None,
) -> np.ndarray:
    try:
        import umap
    except Exception as exc:
        raise ImportError("Install the 'ml' extra to run UMAP") from exc

    cfg = config or ParallelConfig()
    x = standardize_features(features) if standardize else np.asarray(features, dtype=np.float64)
    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=cfg.resolved_workers(maximum=max(1, x.shape[0])),
    ).fit_transform(x)


def feature_matrix(columns: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(col, dtype=np.float64).reshape(-1) for col in columns]
    if not arrays:
        raise ValueError("at least one feature column is required")
    n = arrays[0].shape[0]
    if any(arr.shape[0] != n for arr in arrays):
        raise ValueError("all feature columns must have the same length")
    return np.column_stack(arrays)


__all__ = ["EmbeddingResult", "feature_matrix", "run_pca", "run_umap", "standardize_features"]
