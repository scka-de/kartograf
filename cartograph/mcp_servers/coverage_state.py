"""MCP provider tools exposing Cartograph coverage state."""

from __future__ import annotations

from cartograph.core import storage

from ._common import run_stdio_server


def list_audits(limit: int = 20) -> list[dict]:
    return storage.list_audits(limit)


def get_coverage(audit_id: str) -> dict:
    return storage.load_coverage_report(audit_id).model_dump(mode="json")


def get_uncovered_regions(audit_id: str, top_k: int = 5) -> list[dict]:
    report = storage.load_coverage_report(audit_id)
    return [gap.model_dump(mode="json") for gap in report.gaps[:top_k]]


def get_decision_log(audit_id: str) -> list[dict]:
    report = storage.load_coverage_report(audit_id)
    return [decision.model_dump(mode="json") for decision in report.decisions]


def get_generated_cases(audit_id: str, accepted_only: bool = True) -> list[dict]:
    report = storage.load_coverage_report(audit_id)
    cases = report.generated_cases
    if accepted_only:
        cases = [case for case in cases if case.accepted]
    return [case.model_dump(mode="json") for case in cases]


if __name__ == "__main__":
    run_stdio_server(
        "coverage_state",
        {
            "list_audits": list_audits,
            "get_coverage": get_coverage,
            "get_uncovered_regions": get_uncovered_regions,
            "get_decision_log": get_decision_log,
            "get_generated_cases": get_generated_cases,
        },
    )
