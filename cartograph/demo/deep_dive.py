from __future__ import annotations

import json
from pathlib import Path

from cartograph.agents.generator import run_generator_sync
from cartograph.agents.root import run_audit_sync
from cartograph.core import storage
from cartograph.demo.prepare import _ensure_customer_service_cli_evalset


def run_customer_service_deep_dive(limit: int = 30) -> dict:
    eval_path = _customer_service_eval_path()
    audit_id = run_audit_sync(
        target_agent=_customer_service_target_path(),
        corpus="bitext",
        eval_path=eval_path,
        coverage_threshold=0.75,
        corpus_limit=limit,
    )
    report = storage.load_coverage_report(audit_id)
    if not report.generated_cases and report.regions:
        target_regions = [region.id for region in report.regions[:2]]
        run_generator_sync(
            audit_id,
            target_regions,
            n=2,
            eval_path=eval_path,
        )
        report = storage.load_coverage_report(audit_id)
    payload = _precomputed_payload(report)
    Path("data/precomputed").mkdir(parents=True, exist_ok=True)
    Path("data/precomputed/customer_service_report.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    coverage_before = _coverage_before(report)
    print(f"coverage_before: {coverage_before:.2f}")
    print(f"coverage_after: {report.coverage_score:.2f}")
    print(f"generated_cases: {len(report.generated_cases)}")
    print("decision_log")
    for decision in report.decisions:
        print(f"- {decision.step}: {decision.action} - {decision.reason}")
    print(
        f"pass_rate_before: {report.eval_run_before.pass_rate if report.eval_run_before else 'n/a'}"
    )
    print(f"pass_rate_after: {report.eval_run_after.pass_rate if report.eval_run_after else 'n/a'}")
    blocker = _adk_eval_blocker(report)
    if blocker is not None:
        print(f"adk_eval_blocker: {blocker}")
    rejected = [case for case in report.generated_cases if not case.accepted]
    if rejected:
        print(f"rejection: {rejected[0].id} {rejected[0].validation_status}")
    return payload


def _adk_eval_blocker(report: object) -> str | None:
    messages: list[str] = []
    for attr in ("eval_run_before", "eval_run_after"):
        run = getattr(report, attr, None)
        if run is None or getattr(run, "pass_rate", None) is not None:
            continue
        stderr = str(getattr(run, "stderr", "") or "")
        stdout = str(getattr(run, "stdout", "") or "")
        text = f"{stderr}\n{stdout}"
        if "generativelanguage.googleapis.com" in text and "SERVICE_DISABLED" in text:
            messages.append("Gemini API is disabled or unavailable")
        elif "aiplatform.googleapis.com" in text and "SERVICE_DISABLED" in text:
            messages.append("Vertex Agent Platform API is disabled or unavailable")
        elif "SERVICE_DISABLED" in text:
            messages.append("Google API service is disabled or unavailable")
        elif "API_KEY" in text or "GOOGLE_API_KEY" in text:
            messages.append("GOOGLE_API_KEY is missing or invalid")
        elif "oauth2.googleapis.com" in text:
            messages.append("Google OAuth endpoint is unreachable")
        elif getattr(run, "exit_code", 0):
            messages.append("adk eval returned a non-zero exit code")
    if not messages:
        return None
    return "; ".join(dict.fromkeys(messages))


def _coverage_before(report: object) -> float:
    for decision in getattr(report, "decisions", []):
        if getattr(decision, "action", None) == "invoke_auditor":
            score = getattr(decision, "outputs", {}).get("coverage_score")
            if isinstance(score, int | float):
                return float(score)
    return float(getattr(report, "coverage_score", 0.0))


def _precomputed_payload(report: object) -> dict:
    payload = report.model_dump(mode="json")
    for label in ("eval_run_before", "eval_run_after"):
        run = payload.get(label)
        if not isinstance(run, dict):
            continue
        command = run.get("command")
        if isinstance(command, list) and command:
            command[0] = Path(str(command[0])).name
        run["stdout"] = _summarize_eval_output(str(run.get("stdout", "")))
        run["stderr"] = _summarize_eval_output(str(run.get("stderr", "")))
    return payload


def _summarize_eval_output(text: str) -> str:
    if not text:
        return ""
    markers = [
        "SERVICE_DISABLED",
        "PERMISSION_DENIED",
        "oauth2.googleapis.com",
        "GOOGLE_API_KEY",
        "Timed out after",
    ]
    for marker in markers:
        if marker in text:
            return marker
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if len(first_line) > 240:
        return f"{first_line[:237]}..."
    return first_line


def _customer_service_eval_path() -> str:
    converted = _ensure_customer_service_cli_evalset()
    if converted is not None:
        return str(converted)
    return "data/precomputed/customer-service.evalset.json"


def _customer_service_target_path() -> str:
    module_path = Path("data/adk_samples/python/agents/customer-service/customer_service")
    if module_path.exists():
        return str(module_path)
    sample_path = Path("data/adk_samples/python/agents/customer-service")
    if sample_path.exists():
        return str(sample_path)
    return "python/agents/customer-service"
