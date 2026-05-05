#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
if [[ -x ".venv/bin/python" && "${PYTHON:-}" == "" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

"$PYTHON_BIN" -m pytest
"$PYTHON_BIN" -m ruff check .
"$PYTHON_BIN" -m mypy cartograph/core
"$PYTHON_BIN" -m cartograph.cli version >/tmp/cartograph_version.txt
"$PYTHON_BIN" -m cartograph.cli corpus list >/tmp/cartograph_corpus_list.txt

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

from google.adk.evaluation.eval_set import EvalSet

required = [
    Path("data/precomputed/fleet_benchmark.json"),
    Path("data/precomputed/customer_service_report.json"),
    Path("data/precomputed/customer-service.evalset.json"),
    Path("data/precomputed/customer-service_cartograph_generated.evalset.json"),
]
for path in required:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty required artifact: {path}")

fleet = json.loads(Path("data/precomputed/fleet_benchmark.json").read_text())
agents = fleet.get("agents", [])
if len(agents) != 5:
    raise SystemExit(f"expected 5 fleet agents, found {len(agents)}")
customer = [
    agent for agent in agents if agent.get("target_agent") == "python/agents/customer-service"
]
if not customer:
    raise SystemExit("customer-service entry missing from fleet benchmark")
if customer[0].get("corpus_mode") != "real":
    raise SystemExit("customer-service corpus_mode must be real")

report = json.loads(Path("data/precomputed/customer_service_report.json").read_text())
before = report.get("eval_run_before") or {}
after = report.get("eval_run_after") or {}
if before.get("pass_rate") is None or after.get("pass_rate") is None:
    raise SystemExit("customer_service_report.json has null ADK pass rates")
if report.get("coverage_score", 0.0) < 0.75:
    raise SystemExit("customer_service_report.json coverage_score is below 0.75")

EvalSet.model_validate_json(
    Path("data/precomputed/customer-service_cartograph_generated.evalset.json").read_text()
)

print("submission artifacts OK")
print(f"audit_id {report['audit_id']}")
print(f"coverage {report['coverage_score']}")
print(f"pass_rates {before['pass_rate']} {after['pass_rate']}")
PY

scan_output="$(rg -n '/Users/gustavo|Traceback|AIza|gho_|ghp_|GOOGLE_API_KEY=.' \
  data/precomputed cartograph tests IMPLEMENTATION_PLAN.md README.md .env.example \
  docs/submission_checklist.md docs/devpost_draft.md \
  --glob '!*.pyc' || true)"

unexpected="$(printf "%s\n" "$scan_output" | grep -v 'README.md:.*CARTOGRAPH_RUN_E2E=1 GOOGLE_API_KEY=...' | grep -v 'tests/test_core_adk_eval.py:.*GOOGLE_API_KEY=test-key' | grep -v "docs/submission_checklist.md:.*rg -n" || true)"

if [[ -n "$unexpected" ]]; then
  echo "Unexpected publishability scan matches:" >&2
  printf "%s\n" "$unexpected" >&2
  exit 1
fi

echo "publishability scan OK"
