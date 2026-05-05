import json
from pathlib import Path

import pytest

from cartograph.agents import root
from cartograph.agents.common import log_decision, observe_decisions
from cartograph.agents.root import decide_generate_target, write_after_eval_file
from cartograph.core import storage
from cartograph.core.models import EvalRunResult, Gap, GeneratedCase


def test_decide_generate_target_uses_top_ranked_gaps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage.create_audit("audit_test", "target", "bitext")
    storage.save_gaps(
        "audit_test",
        [
            Gap(region_id="region_01", risk_score=0.8, rank=1),
            Gap(region_id="region_02", risk_score=0.7, rank=2),
            Gap(region_id="region_03", risk_score=0.1, rank=3),
        ],
    )
    assert decide_generate_target("audit_test", 2) == {"region_ids": ["region_01", "region_02"]}


def test_decision_observer_streams_logged_decisions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage.create_audit("audit_test", "target", "bitext")
    seen: list[tuple[str, str]] = []

    with observe_decisions(lambda audit_id, decision: seen.append((audit_id, decision.action))):
        log_decision("audit_test", "ingest", {}, {}, "loaded corpus")

    log_decision("audit_test", "invoke_mapper", {}, {}, "mapped corpus")

    assert seen == [("audit_test", "ingest")]


def test_write_after_eval_file_merges_current_adk_test_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage.create_audit("audit_test", "target", "bitext")
    storage.save_generated_cases(
        "audit_test",
        [
            GeneratedCase(
                id="gen_1",
                region_id="region_00",
                content="new query",
                raw={"expected_final_response": "reference"},
                validation_status="accepted",
                accepted=True,
            )
        ],
    )
    eval_path = tmp_path / "simple.test.json"
    eval_path.write_text(json.dumps([{"query": "old", "reference": "old ref"}]))
    output = write_after_eval_file("audit_test", str(eval_path))
    payload = json.loads((tmp_path / output).read_text())
    assert [case["query"] for case in payload] == ["old", "new query"]


def test_write_after_eval_file_merges_old_cli_evalset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage.create_audit("audit_test", "target", "bitext")
    storage.save_generated_cases(
        "audit_test",
        [
            GeneratedCase(
                id="gen_1",
                region_id="region_00",
                content="new query",
                raw={"expected_final_response": "reference"},
                validation_status="accepted",
                accepted=True,
            )
        ],
    )
    eval_path = tmp_path / "simple.evalset.json"
    eval_path.write_text(json.dumps([{"name": "old", "data": [{"query": "old"}]}]))
    output = write_after_eval_file("audit_test", str(eval_path))
    payload = json.loads((tmp_path / output).read_text())
    assert payload[1]["name"] == "gen_1"
    assert payload[1]["data"][0]["query"] == "new query"


@pytest.mark.asyncio
async def test_run_audit_reaudits_merged_generated_evalset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    eval_path = tmp_path / "simple.evalset.json"
    eval_path.write_text(
        json.dumps(
            {
                "eval_set_id": "simple",
                "eval_cases": [
                    {
                        "eval_id": "old",
                        "conversation": [{"role": "user", "content": "old query"}],
                    }
                ],
            }
        )
    )
    auditor_paths: list[str] = []

    def fake_run_adk_eval(agent_path, evalset_path, label, timeout_seconds=600):
        return EvalRunResult(
            label=label,
            command=["adk", "eval", agent_path, evalset_path],
            exit_code=0,
            pass_count=1,
            fail_count=0,
            pass_rate=1.0,
            stdout="1/1 passed",
            stderr="",
            duration_seconds=0.01,
        )

    async def fake_run_mapper(audit_id, corpus, corpus_limit, query_args):
        return {"region_count": 1}

    async def fake_run_auditor(audit_id, current_eval_path):
        auditor_paths.append(current_eval_path)
        return {"coverage_score": 0.4}

    async def fake_run_generator(audit_id, region_ids, eval_path=None):
        storage.save_generated_cases(
            audit_id,
            [
                GeneratedCase(
                    id="gen_1",
                    region_id="region_00",
                    content="new query",
                    raw={"expected_final_response": "reference"},
                    validation_status="accepted",
                    accepted=True,
                )
            ],
        )
        return {"accepted": 1}

    decisions = iter(
        [
            {"terminate": False, "reason": "coverage below threshold"},
            {"terminate": True, "reason": "coverage above threshold"},
        ]
    )
    monkeypatch.setattr(root, "run_adk_eval", fake_run_adk_eval)
    monkeypatch.setattr(root, "run_mapper", fake_run_mapper)
    monkeypatch.setattr(root, "run_auditor", fake_run_auditor)
    monkeypatch.setattr(root, "run_generator", fake_run_generator)
    monkeypatch.setattr(root, "decide_terminate", lambda audit_id, threshold=0.75: next(decisions))
    monkeypatch.setattr(
        root,
        "decide_generate_target",
        lambda audit_id, k=2: {"region_ids": ["region_00"]},
    )

    audit_id = await root.run_audit("target", "bitext", str(eval_path))

    assert auditor_paths[0] == str(eval_path)
    assert auditor_paths[1] == str(storage.audit_dir(audit_id) / "eval_run_after.evalset.json")
    merged = json.loads(Path(auditor_paths[1]).read_text())
    assert [case["eval_id"] for case in merged["eval_cases"]] == ["old", "gen_1"]
