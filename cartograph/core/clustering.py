from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


@dataclass
class ProjectionReducer:
    """SVD-based 2D fallback when UMAP is unavailable."""

    components: np.ndarray
    mean: np.ndarray
    target_dim: int

    def transform(self, embeddings: np.ndarray) -> NDArray[np.float64]:
        arr = np.asarray(embeddings, dtype=float)
        if arr.size == 0:
            return cast(NDArray[np.float64], np.empty((0, self.target_dim), dtype=float))
        centered = arr - self.mean
        projected = centered @ self.components.T
        if projected.shape[1] < self.target_dim:
            padding = np.zeros((projected.shape[0], self.target_dim - projected.shape[1]))
            projected = np.hstack([projected, padding])
        return cast(NDArray[np.float64], projected[:, : self.target_dim])


def fit_visualization_reducer(embeddings: np.ndarray) -> Any:
    """Fit a UMAP reducer to 2D, falling back to SVD if UMAP is missing or fails."""
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        reducer.fit(embeddings)
        return reducer
    except Exception:
        return _fit_projection_reducer(embeddings, 2)


def _fit_projection_reducer(embeddings: np.ndarray, target_dim: int) -> ProjectionReducer:
    arr = np.asarray(embeddings, dtype=float)
    if arr.size == 0:
        return ProjectionReducer(
            np.zeros((target_dim, target_dim)), np.zeros(target_dim), target_dim
        )
    mean = arr.mean(axis=0)
    centered = arr - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:target_dim]
    if components.shape[0] < target_dim:
        padding = np.zeros((target_dim - components.shape[0], arr.shape[1]))
        components = np.vstack([components, padding])
    return ProjectionReducer(components=components, mean=mean, target_dim=target_dim)
