#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
if [[ -x ".venv/bin/python" && "${PYTHON:-}" == "" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

"$PYTHON_BIN" -m pytest
"$PYTHON_BIN" -m ruff check .
"$PYTHON_BIN" -m mypy cartograph/core
"$PYTHON_BIN" -m cartograph.cli version >/tmp/kartograf_version.txt

HERO_HTMLS=(
  "docs/figures/kartograf_map_customer_service.html:python/agents/customer-service"
  "docs/figures/kartograf_map_financial_advisor.html:python/agents/financial-advisor"
)
for entry in "${HERO_HTMLS[@]}"; do
  HERO_HTML="${entry%%:*}"
  HERO_TARGET="${entry##*:}"
  if [[ ! -f "$HERO_HTML" ]]; then
    echo "missing hero artifact: $HERO_HTML" >&2
    echo "regenerate with: $PYTHON_BIN -m cartograph.cli map --target data/adk_samples/$HERO_TARGET --output $HERO_HTML" >&2
    exit 1
  fi
  LATEST_SRC=$(find cartograph -type f -name '*.py' -newer "$HERO_HTML" 2>/dev/null | head -1 || true)
  if [[ -n "$LATEST_SRC" ]]; then
    echo "hero artifact stale (newer source: $LATEST_SRC -> $HERO_HTML)" >&2
    echo "regenerate with: $PYTHON_BIN -m cartograph.cli map --target data/adk_samples/$HERO_TARGET --output $HERO_HTML" >&2
    exit 1
  fi
  echo "hero artifact OK ($HERO_HTML)"
done

echo "submission OK"
