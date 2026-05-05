#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
if [[ -x ".venv/bin/python" && "${PYTHON:-}" == "" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/data/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/data/cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

"$PYTHON_BIN" -m cartograph.cli demo prepare
"$PYTHON_BIN" -m cartograph.cli demo run customer-service | tee /tmp/cartograph_demo_output.txt

latest_audit_id="$("$PYTHON_BIN" - <<'PY'
from cartograph.core import storage
print(storage.latest_audit_id() or "")
PY
)"

if [[ -z "$latest_audit_id" ]]; then
  echo "No latest audit found" >&2
  exit 1
fi

"$PYTHON_BIN" -m cartograph.cli demo visualize "$latest_audit_id"

required_files=(
  "data/precomputed/fleet_benchmark.json"
  "data/precomputed/customer_service_report.json"
  "data/audits/$latest_audit_id/viz/regions.png"
  "data/audits/$latest_audit_id/viz/coverage.png"
  "data/audits/$latest_audit_id/viz/before_after.png"
  "data/audits/$latest_audit_id/reducer_30d.joblib"
  "data/audits/$latest_audit_id/reducer_2d.joblib"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing required file: $file" >&2
    exit 1
  fi
done

for needle in coverage_before: coverage_after: generated_cases: decision_log pass_rate_before: pass_rate_after:; do
  if ! grep -q "$needle" /tmp/cartograph_demo_output.txt; then
    echo "Missing required stdout string: $needle" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("data/precomputed/fleet_benchmark.json").read_text())
matches = [a for a in payload["agents"] if a["target_agent"] == "python/agents/customer-service"]
if not matches:
    raise SystemExit("customer-service entry missing from fleet_benchmark.json")
if matches[0]["corpus_mode"] != "real":
    raise SystemExit("customer-service corpus_mode must be real")
PY

echo "Cartograph demo acceptance passed for $latest_audit_id"
