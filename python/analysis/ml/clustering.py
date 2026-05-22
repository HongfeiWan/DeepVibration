"""Reusable clustering helpers for cleaned parameter matrices."""

from __future__ import annotations

import numpy as np


def run_hdbscan(
    features: np.ndarray,
    *,
    min_cluster_size: int = 20,
    min_samples: int | None = None,
    metric: str = "euclidean",
) -> np.ndarray:
    try:
        import hdbscan
    except Exception as exc:
        raise ImportError("Install the 'ml' extra to run HDBSCAN") from exc

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    return clusterer.fit_predict(np.asarray(features, dtype=np.float64))


def run_gmm(
    features: np.ndarray,
    *,
    n_components: int = 2,
    covariance_type: str = "full",
    random_state: int | None = 42,
) -> np.ndarray:
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as exc:
        raise ImportError("Install the 'ml' extra to run GaussianMixture") from exc

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    return model.fit_predict(np.asarray(features, dtype=np.float64))


def ledoit_wolf_covariance(features: np.ndarray) -> tuple[np.ndarray, float]:
    try:
        from sklearn.covariance import LedoitWolf
    except Exception as exc:
        raise ImportError("Install the 'ml' extra to run LedoitWolf covariance") from exc

    model = LedoitWolf().fit(np.asarray(features, dtype=np.float64))
    return model.covariance_, float(model.shrinkage_)


__all__ = ["ledoit_wolf_covariance", "run_gmm", "run_hdbscan"]
