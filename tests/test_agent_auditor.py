import numpy as np

from cartograph.agents.auditor import (
    _reuse_exact_corpus_vectors,
    assign_evals_to_regions,
    flag_redundancies,
)
from cartograph.core import storage
from cartograph.core.models import CorpusItem, EvalCase


def test_auditor_assigns_eval_to_nearest_region(tiny_eval_suite, tiny_regions):
    suspicious = assign_evals_to_regions(tiny_eval_suite, tiny_regions)
    assert suspicious == 0
    assert tiny_eval_suite[0].region_id == "region_00"
    assert tiny_eval_suite[0].assignment_distance == 0.0


def test_auditor_flags_redundancies(tiny_eval_suite):
    duplicate = tiny_eval_suite[0].model_copy(update={"id": "eval_2"})
    redundant = flag_redundancies([tiny_eval_suite[0], duplicate])
    assert redundant[0].case_id == "eval_2"


def test_auditor_reuses_exact_corpus_vectors_for_generated_exemplars(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage.create_audit("audit_test", "target", "bitext")
    storage.save_corpus_items(
        "audit_test",
        [CorpusItem(id="c_1", source="test", text="cancel order 123")],
    )
    storage.save_corpus_embeddings(
        "audit_test",
        full=np.asarray([[1.0, 2.0, 3.0]]),
        reduced_30d=np.asarray([[1.0] + [0.0] * 29]),
        reduced_2d=np.asarray([[1.0, 0.0]]),
    )
    cases = [EvalCase(id="eval_1", source_path="eval.json", content="cancel order 123")]
    full = np.asarray([[9.0, 9.0, 9.0]])
    reduced = np.asarray([[0.0, 1.0] + [0.0] * 28])

    _reuse_exact_corpus_vectors("audit_test", cases, full, reduced)

    assert full.tolist() == [[1.0, 2.0, 3.0]]
    assert reduced.tolist() == [[1.0] + [0.0] * 29]
