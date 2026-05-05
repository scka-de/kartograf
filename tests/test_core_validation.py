import numpy as np

from cartograph.core.models import EvalCase, GeneratedCase
from cartograph.core.validation import validate_generated_case


def test_validation_rejects_redundant_candidate():
    existing = [EvalCase(id="eval_1", source_path="x", content="x", embedding=[1.0] + [0.0] * 767)]
    candidate = GeneratedCase(
        id="gen_1",
        region_id="region_00",
        content="x",
        embedding=[1.0] + [0.0] * 767,
        embedding_30d=[1.0] + [0.0] * 29,
        validation_status="accepted",
        accepted=True,
    )
    result = validate_generated_case(candidate, existing, np.asarray([[1.0] + [0.0] * 29]))
    assert result.validation_status == "rejected_redundant"
    assert not result.accepted


def test_validation_rejects_off_corpus_candidate():
    existing = [
        EvalCase(id="eval_1", source_path="x", content="x", embedding=[0.0, 1.0] + [0.0] * 766)
    ]
    candidate = GeneratedCase(
        id="gen_1",
        region_id="region_00",
        content="x",
        embedding=[1.0] + [0.0] * 767,
        embedding_30d=[0.0, 1.0] + [0.0] * 28,
        validation_status="accepted",
        accepted=True,
    )
    result = validate_generated_case(candidate, existing, np.asarray([[1.0] + [0.0] * 29]))
    assert result.validation_status == "rejected_off_corpus"
    assert not result.accepted
