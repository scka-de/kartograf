from __future__ import annotations

import hashlib
from typing import cast

import numpy as np
from numpy.typing import NDArray


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def l2_normalize(matrix: NDArray[np.floating] | NDArray[np.integer]) -> NDArray[np.float64]:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return cast(NDArray[np.float64], arr if norm == 0 else arr / norm)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return cast(NDArray[np.float64], arr / norms)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = l2_normalize(np.asarray(a, dtype=float))
    bb = l2_normalize(np.asarray(b, dtype=float))
    return float(np.dot(aa, bb))


def cosine_distance(a: np.ndarray, b: np.ndarray, clamp: bool = False) -> float:
    distance = 1.0 - cosine_similarity(a, b)
    if clamp:
        return max(0.0, min(1.0, distance))
    return float(distance)


def deterministic_embedding(text: str, dim: int = 768) -> NDArray[np.float64]:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim)
    return l2_normalize(vec)


def stable_cache_key(source: str, texts: list[str]) -> str:
    joined = "\n".join(texts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    return f"{source}_{digest}"
