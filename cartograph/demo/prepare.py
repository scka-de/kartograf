from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from cartograph.agents.root import run_audit_sync
from cartograph.core import storage
from cartograph.mcp_servers import (
    corpus_reader_bitext,
    corpus_reader_github,
    corpus_reader_stackexchange,
)

FLEET = [
    {
        "target_agent": "python/agents/customer-service",
        "corpus": "bitext",
        "query_args": {},
    },
    {
        "target_agent": "python/agents/software-bug-assistant",
        "corpus": "github",
        "query_args": {"owner": "psf", "repo": "requests"},
    },
    {
        "target_agent": "python/agents/financial-advisor",
        "corpus": "stackexchange",
        "query_args": {"site": "money"},
    },
    {
        "target_agent": "python/agents/travel-concierge",
        "corpus": "stackexchange",
        "query_args": {"site": "travel"},
    },
    {
        "target_agent": "python/agents/data-science",
        "corpus": "stackexchange",
        "query_args": {"site": "stackoverflow", "tag": "pandas"},
    },
]


def prepare_fleet_benchmark(limit: int = 30) -> dict:
    Path("data/precomputed").mkdir(parents=True, exist_ok=True)
    agents = []
    for entry in FLEET:
        audit_id = run_audit_sync(
            target_agent=_target_path_for(entry["target_agent"]),
            corpus=entry["corpus"],
            eval_path=_eval_path_for(entry["target_agent"]),
            coverage_threshold=0.75,
            corpus_limit=limit,
            query_args=entry["query_args"],
        )
        report = storage.load_coverage_report(audit_id)
        row = {
            "target_agent": entry["target_agent"],
            "corpus_mode": _corpus_mode(entry["corpus"], entry["query_args"]),
            "corpus_size": report.corpus_size,
            "noise_fraction": report.noise_fraction,
            "coverage_score": report.coverage_score,
            "region_count": len(report.regions),
            "audit_id": audit_id,
        }
        if row["corpus_mode"] == "mocked":
            row["corpus_source_failure_reason"] = (
                "using bundled deterministic fixture in local demo mode"
            )
        agents.append(row)
    payload = {"generated_at": datetime.now(UTC).isoformat(), "agents": agents}
    Path("data/precomputed/fleet_benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _corpus_mode(corpus: str, query_args: dict) -> str:
    if corpus == "bitext":
        return corpus_reader_bitext.get_mode()
    if corpus == "github":
        return corpus_reader_github.get_mode()
    if corpus == "stackexchange":
        return corpus_reader_stackexchange.get_mode()
    return "unknown"


def _eval_path_for(target_agent: str) -> str:
    if target_agent == "python/agents/customer-service":
        converted = _ensure_customer_service_cli_evalset()
        if converted is not None:
            return str(converted)
    return f"data/precomputed/{target_agent.split('/')[-1]}.evalset.json"


def _target_path_for(target_agent: str) -> str:
    if target_agent == "python/agents/customer-service":
        module_path = Path("data/adk_samples/python/agents/customer-service/customer_service")
        if module_path.exists():
            return str(module_path)
    sample_path = Path("data/adk_samples") / target_agent
    if sample_path.exists():
        return str(sample_path)
    return target_agent


def _ensure_customer_service_cli_evalset() -> Path | None:
    source = Path("data/adk_samples/python/agents/customer-service/eval/eval_data/simple.test.json")
    if not source.exists():
        return None
    target = Path("data/precomputed/customer-service.evalset.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text())
    converted = {
        "eval_set_id": "customer_service_cartograph",
        "name": "customer_service_cartograph",
        "eval_cases": [
            _sample_item_to_eval_case(item, index) for index, item in enumerate(payload, start=1)
        ],
    }
    target.write_text(json.dumps(converted, indent=2) + "\n")
    return target


def _sample_item_to_eval_case(item: dict, index: int) -> dict:
    return {
        "eval_id": f"customer_service_{index:02d}",
        "conversation": [
            {
                "user_content": {
                    "parts": [{"text": item.get("query", "")}],
                    "role": "user",
                },
                "final_response": {
                    "parts": [{"text": item.get("reference", "")}],
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
