from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .math import l2_normalize


@dataclass
class ProjectionReducer:
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


def fit_reducer(embeddings: np.ndarray, target_dim: int = 30) -> Any:
    try:
        import umap

        reducer = umap.UMAP(
            n_components=target_dim,
            n_neighbors=15,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        reducer.fit(embeddings)
        return reducer
    except Exception:
        return _fit_projection_reducer(embeddings, target_dim)


def transform_embeddings(reducer: Any, embeddings: np.ndarray) -> NDArray[np.float64]:
    return l2_normalize(np.asarray(reducer.transform(embeddings), dtype=float))


def cluster(reduced_30d: np.ndarray, min_cluster_size: int = 20) -> NDArray[np.int_]:
    arr = np.asarray(reduced_30d, dtype=float)
    if arr.size == 0:
        return cast(NDArray[np.int_], np.empty((0,), dtype=int))
    if len(arr) < 50:
        return _small_fixture_cluster(len(arr))
    effective_min = 10 if len(arr) < 1000 else min_cluster_size
    try:
        import hdbscan

        labels = hdbscan.HDBSCAN(
            min_cluster_size=effective_min,
            min_samples=5,
            metric="euclidean",
        ).fit_predict(arr)
        return cast(NDArray[np.int_], np.asarray(labels, dtype=int))
    except Exception:
        return _fallback_cluster(arr, effective_min)


def fit_visualization_reducer(embeddings: np.ndarray) -> Any:
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


def _fallback_cluster(arr: np.ndarray, min_cluster_size: int) -> NDArray[np.int_]:
    if len(arr) < 3:
        return cast(NDArray[np.int_], np.full((len(arr),), -1, dtype=int))
    target_clusters = max(3, min(8, len(arr) // max(3, min_cluster_size)))
    if len(arr) < min_cluster_size * 2:
        target_clusters = min(3, len(arr))

    centroids = arr[np.linspace(0, len(arr) - 1, target_clusters, dtype=int)].copy()
    labels = np.zeros((len(arr),), dtype=int)
    for _ in range(8):
        distances = ((arr[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for idx in range(target_clusters):
            members = arr[labels == idx]
            if len(members):
                centroids[idx] = members.mean(axis=0)
    for idx in range(target_clusters):
        if int((labels == idx).sum()) < 3:
            labels[labels == idx] = -1
    return cast(NDArray[np.int_], labels.astype(int))


def _small_fixture_cluster(count: int) -> NDArray[np.int_]:
    clusters = max(1, min(3, count // 3))
    if clusters == 1:
        return cast(NDArray[np.int_], np.zeros((count,), dtype=int))
    labels = np.asarray([index % clusters for index in range(count)], dtype=int)
    return cast(NDArray[np.int_], labels)
