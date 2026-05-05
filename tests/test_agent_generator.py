import json

from cartograph.agents.generator import _local_candidates, format_for_target
from cartograph.core import storage
from cartograph.core.models import GeneratedCase


def test_generator_local_candidates_include_requested_count_plus_control():
    candidates = _local_candidates("region_00", "Refund", ["I need a refund"], 3)
    assert len(candidates) == 4
    assert "mortgage" in candidates[-1]


def test_generator_local_candidates_uses_label_when_no_exemplars():
    candidates = _local_candidates("region_00", "Refund", [], 1)
    assert "refund" in candidates[0].lower()


def test_format_for_target_replaces_generated_evalset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage.create_audit("audit_test", "target", "bitext")
    eval_path = tmp_path / "suite.evalset.json"
    eval_path.write_text(json.dumps({"eval_cases": []}))
    output = format_for_target(
        "audit_test",
        [
            GeneratedCase(
                id="gen_1",
                region_id="region_00",
                content="new query",
                raw={
                    "eval_id": "gen_1",
                    "conversation": [{"role": "user", "content": "new query"}],
                },
                validation_status="accepted",
                accepted=True,
            )
        ],
        str(eval_path),
    )
    format_for_target(
        "audit_test",
        [
            GeneratedCase(
                id="gen_2",
                region_id="region_00",
                content="newer query",
                raw={
                    "eval_id": "gen_2",
                    "conversation": [{"role": "user", "content": "newer query"}],
                },
                validation_status="accepted",
                accepted=True,
            )
        ],
        str(eval_path),
    )
    payload = json.loads((tmp_path / output).read_text())
    assert [case["eval_id"] for case in payload["eval_cases"]] == ["gen_2"]
