from __future__ import annotations

import numpy as np

from .math import cosine_distance, cosine_similarity
from .models import EvalCase, GeneratedCase


def validate_generated_case(
    candidate: GeneratedCase,
    existing_evals: list[EvalCase],
    exemplar_embeddings_30d: np.ndarray,
    redundancy_threshold: float = 0.85,
    off_corpus_distance_threshold: float = 0.42,
) -> GeneratedCase:
    if candidate.embedding is None or candidate.embedding_30d is None:
        raise ValueError("candidate must have both 768d and 30d embeddings")

    max_similarity = 0.0
    duplicate_of = None
    for eval_case in existing_evals:
        if eval_case.embedding is None:
            continue
        similarity = cosine_similarity(
            np.asarray(candidate.embedding), np.asarray(eval_case.embedding)
        )
        if similarity > max_similarity:
            max_similarity = similarity
            duplicate_of = eval_case.id
    if max_similarity > redundancy_threshold:
        return candidate.model_copy(
            update={
                "validation_status": "rejected_redundant",
                "validation_details": {
                    "max_similarity": max_similarity,
                    "duplicate_of_case_id": duplicate_of,
                },
                "accepted": False,
            }
        )

    if exemplar_embeddings_30d.size == 0:
        min_distance = 1.0
    else:
        candidate_30d = np.asarray(candidate.embedding_30d)
        min_distance = min(
            cosine_distance(candidate_30d, exemplar, clamp=True)
            for exemplar in exemplar_embeddings_30d
        )
    if min_distance > off_corpus_distance_threshold:
        return candidate.model_copy(
            update={
                "validation_status": "rejected_off_corpus",
                "validation_details": {"min_exemplar_distance": min_distance},
                "accepted": False,
            }
        )

    return candidate.model_copy(
        update={
            "validation_status": "accepted",
            "validation_details": {
                "max_similarity": max_similarity,
                "min_exemplar_distance": min_distance,
            },
            "accepted": True,
        }
    )
