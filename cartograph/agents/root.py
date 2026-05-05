from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from cartograph.core import storage
from cartograph.core.adk_eval import run_adk_eval

from .auditor import run_auditor
from .common import log_decision
from .generator import run_generator
from .mapper import run_mapper

ROOT_PROMPT = """
You are Cartograph, an autonomous agent that audits eval suites.

You have three sub-agents available:
- mapper: maps the real-world corpus into semantic regions
- auditor: scores coverage and ranks gaps
- generator: produces new eval cases for uncovered regions

You also have these decision functions:
- decide_generate_target(audit_id) -> dict
- decide_terminate(audit_id) -> bool

Your workflow for any audit:

1. Call transfer_to_agent("mapper") to map the corpus.
2. Call transfer_to_agent("auditor") to score coverage.
3. If coverage >= COVERAGE_THRESHOLD (default 0.75), call decide_terminate. Skip to step 7.
4. Call decide_generate_target to choose top-K uncovered regions to target.
5. Call transfer_to_agent("generator") with those region_ids.
6. Call transfer_to_agent("auditor") to re-audit. Then return to step 3.
7. Generate the final report using format_final_report.

At every transition, log a Decision row with the action, inputs, outputs, and your one-line reason.

Loop limit: max 3 generation rounds total. After 3 rounds, terminate regardless of coverage.

Be decisive. Do not ask for clarification. Do not propose alternatives. Execute the workflow.
"""


async def run_audit(
    target_agent: str,
    corpus: str,
    eval_path: str,
    coverage_threshold: float = 0.75,
    corpus_limit: int = 1000,
    query_args: dict | None = None,
) -> str:
    audit_id = f"audit_{uuid.uuid4().hex[:12]}"
    storage.create_audit(audit_id, target_agent, corpus)
    before = run_adk_eval(target_agent, eval_path, "before", timeout_seconds=120)
    storage.save_eval_run(audit_id, before)
    await run_mapper(audit_id, corpus, corpus_limit, query_args or {})
    current_eval_path = eval_path
    await run_auditor(audit_id, current_eval_path)
    rounds = 0
    while rounds < 3:
        decision = decide_terminate(audit_id, coverage_threshold)
        log_decision(
            audit_id,
            "decide_terminate" if decision["terminate"] else "decide_continue",
            {"threshold": coverage_threshold},
            decision,
            decision["reason"],
        )
        if decision["terminate"]:
            break
        target = decide_generate_target(audit_id, 2)
        if not target["region_ids"]:
            break
        await run_generator(audit_id, target["region_ids"], eval_path=eval_path)
        current_eval_path = write_after_eval_file(audit_id, eval_path)
        await run_auditor(audit_id, current_eval_path)
        rounds += 1
    after_eval_path = (
        current_eval_path
        if current_eval_path != eval_path
        else write_after_eval_file(audit_id, eval_path)
    )
    after = run_adk_eval(target_agent, after_eval_path, "after", timeout_seconds=120)
    storage.save_eval_run(audit_id, after)
    final = storage.load_coverage_report(audit_id)
    storage.update_audit_summary(
        audit_id, final.coverage_score, final.noise_fraction, final.warnings, completed=True
    )
    log_decision(audit_id, "format_final_report", {}, {"audit_id": audit_id}, "final report ready")
    return audit_id


def decide_generate_target(audit_id: str, k: int = 2) -> dict:
    report = storage.load_coverage_report(audit_id)
    return {"region_ids": [gap.region_id for gap in report.gaps[:k]]}


def decide_terminate(audit_id: str, threshold: float = 0.75) -> dict:
    report = storage.load_coverage_report(audit_id)
    if report.coverage_score >= threshold:
        return {
            "terminate": True,
            "reason": f"coverage {report.coverage_score:.2f} >= {threshold:.2f}",
        }
    if not report.gaps:
        return {"terminate": True, "reason": "no uncovered gaps remain"}
    return {"terminate": False, "reason": f"coverage {report.coverage_score:.2f} below threshold"}


def format_final_report(audit_id: str):
    return storage.load_coverage_report(audit_id)


def write_after_eval_file(audit_id: str, eval_path: str) -> str:
    source = Path(eval_path)
    output_name = (
        "eval_run_after.test.json"
        if source.name.endswith(".test.json")
        else "eval_run_after.evalset.json"
    )
    output = storage.audit_dir(audit_id) / output_name
    report = storage.load_coverage_report(audit_id)
    generated = [case for case in report.generated_cases if case.accepted]
    if source.exists():
        payload = json.loads(source.read_text())
    else:
        payload = {"eval_set_id": source.stem, "name": source.stem, "eval_cases": []}

    if isinstance(payload, list):
        if _is_old_evalset(payload):
            converter = _generated_to_old_eval_case
        else:
            converter = _generated_to_test_case
        payload = payload + [converter(case) for case in generated]
    else:
        payload["eval_cases"] = payload.get("eval_cases", []) + [
            _generated_to_new_eval_case(case) for case in generated
        ]
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return str(output)


def _generated_to_test_case(case) -> dict:
    return {
        "query": case.content,
        "expected_tool_use": [],
        "reference": case.raw.get(
            "expected_final_response",
            "Handle the user request according to policy.",
        ),
    }


def _generated_to_old_eval_case(case) -> dict:
    return {
        "name": case.id,
        "data": [_generated_to_test_case(case)],
    }


def _generated_to_new_eval_case(case) -> dict:
    return {
        "eval_id": case.id,
        "conversation": [
            {
                "user_content": {
                    "parts": [{"text": case.content}],
                    "role": "user",
                },
                "final_response": {
                    "parts": [
                        {
                            "text": case.raw.get(
                                "expected_final_response",
                                "Handle the user request according to policy.",
                            )
                        }
                    ],
                    "role": "model",
                },
            }
        ],
        "session_input": {
            "app_name": "customer_service",
            "user_id": "user",
            "state": {},
        },
    }


def _is_old_evalset(payload: list) -> bool:
    return bool(payload) and isinstance(payload[0], dict) and "data" in payload[0]


def run_audit_sync(*args, **kwargs) -> str:
    return asyncio.run(run_audit(*args, **kwargs))
