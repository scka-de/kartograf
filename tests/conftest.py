from __future__ import annotations

import numpy as np
import pytest

from cartograph.core.models import CorpusItem, EvalCase, Region


@pytest.fixture
def tiny_regions() -> list[Region]:
    return [
        Region(
            id="region_00",
            label="Refund",
            centroid=[1.0] + [0.0] * 29,
            density=0.6,
            heterogeneity=0.5,
            member_count=10,
            eval_case_ids=["eval_1"],
        ),
        Region(
            id="region_01",
            label="Shipping",
            centroid=[0.0, 1.0] + [0.0] * 28,
            density=0.4,
            heterogeneity=0.25,
            member_count=8,
        ),
    ]


@pytest.fixture
def tiny_corpus() -> list[CorpusItem]:
    return [
        CorpusItem(id=f"c_{i}", source="test", text=f"refund request {i}") for i in range(10)
    ] + [CorpusItem(id=f"s_{i}", source="test", text=f"shipping status {i}") for i in range(10)]


@pytest.fixture
def tiny_eval_suite() -> list[EvalCase]:
    return [
        EvalCase(
            id="eval_1",
            source_path="fixture.evalset.json",
            content="refund request",
            embedding=[1.0] + [0.0] * 767,
            embedding_30d=[1.0] + [0.0] * 29,
        )
    ]


@pytest.fixture
def eval_embeddings_30d() -> np.ndarray:
    return np.asarray([[1.0] + [0.0] * 29])
