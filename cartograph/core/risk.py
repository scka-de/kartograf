from __future__ import annotations

from typing import Any

import numpy as np

from .math import cosine_distance
from .models import Region


def compute_risk(region: Region, eval_embeddings_30d: np.ndarray) -> dict[str, Any]:
    if len(region.centroid) != 30:
        raise ValueError("region centroid must be 30d")
    if eval_embeddings_30d.size and eval_embeddings_30d.shape[1] != 30:
        raise ValueError("eval embeddings must be 30d")

    if eval_embeddings_30d.size == 0:
        distance = 1.0
    else:
        centroid = np.asarray(region.centroid, dtype=float)
        distance = min(
            cosine_distance(centroid, eval_embedding, clamp=True)
            for eval_embedding in eval_embeddings_30d
        )
    heterogeneity = max(0.0, min(1.0, region.heterogeneity))
    density = max(0.0, min(1.0, region.density))
    risk_score = density * heterogeneity * distance
    return {
        "risk_score": float(risk_score),
        "components": {
            "density": float(density),
            "heterogeneity": float(heterogeneity),
            "distance": float(distance),
        },
    }
