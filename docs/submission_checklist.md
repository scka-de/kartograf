# Cartograph Submission Checklist

Use this checklist for the final GitHub/Devpost submission pass. The code and
precomputed artifacts are committed; the remaining work is external publication
and recording.

## Preflight

1. Confirm `.env` is configured for a live Google runtime:

   ```bash
   GOOGLE_GENAI_USE_VERTEXAI=1
   GOOGLE_CLOUD_PROJECT=kartograf-495419
   GOOGLE_CLOUD_LOCATION=us-central1
   ```

   For Gemini Developer API instead, use `GOOGLE_API_KEY` and
   `GOOGLE_GENAI_USE_VERTEXAI=0`.

2. Run the default quality gates:

   ```bash
   .venv/bin/python -m pytest
   .venv/bin/ruff check .
   .venv/bin/mypy cartograph/core
   ```

3. Run the live pre-submission smoke test:

   ```bash
   GOOGLE_GENAI_USE_VERTEXAI=1 CARTOGRAPH_RUN_E2E=1 .venv/bin/python -m pytest -m e2e
   ```

4. Verify the committed report still has live ADK pass-rate values:

   ```bash
   .venv/bin/python - <<'PY'
   import json
   from pathlib import Path

   report = json.loads(Path("data/precomputed/customer_service_report.json").read_text())
   print(report["audit_id"])
   print(report["coverage_score"])
   print(report["eval_run_before"]["pass_rate"], report["eval_run_after"]["pass_rate"])
   PY
   ```

   Both pass-rate values must be non-null for the submission artifact.

## Recording Arc

The recording should follow the proposal's four-minute arc in
`docs/cartograph_proposal.md` section 9.

| Time | Screen | Talk Track |
|---|---|---|
| 0:00-0:30 | README plus proposal/spec links | Cartograph measures whether eval suites look in the right places. ADK samples are educational references, so the demo measures a universal eval-suite gap rather than criticizing the samples. |
| 0:30-1:00 | `data/precomputed/fleet_benchmark.json` or `cartograph demo prepare` output | Five ADK sample agents are mapped against domain-matched corpora. Call out `corpus_mode`; if any entry is mocked, acknowledge fallback mode explicitly. |
| 1:00-2:00 | `cartograph demo run customer-service` output | Show coverage before/after lines and the live decision log: ingest, mapper, auditor, continue/generate decisions, re-audit, final report. |
| 2:00-2:40 | Generated cases in `data/precomputed/customer-service_cartograph_generated.evalset.json` | Show accepted generated cases and at least one rejection event. Emphasize corpus-grounded synthesis with redundancy/off-corpus validation. |
| 2:40-3:30 | `data/precomputed/customer_service_report.json` | Show pass-rate comparison and coverage comparison. Use the framing line: "the agent did not get worse; the test got more honest." |
| 3:30-4:00 | Visualization PNGs under `data/audits/<audit_id>/viz/` | Close with: "Cartograph does not tell you whether your agent is correct everywhere. It tells you whether your tests are looking in the right places." |

## Recording Commands

For a live recording with warm caches:

```bash
bash scripts/run_demo.sh
```

If this runs without external Google/Vertex access, it can still pass fallback
acceptance but may write `adk_eval_blocker` and null pass rates into
`data/precomputed/customer_service_report.json`. Before publishing, restore the
Vertex-verified artifact or rerun the live e2e smoke test and commit the updated
report.

## GitHub And Devpost

1. Confirm the repository is intended to be public. It is private until explicitly
   changed for submission.
2. Make the GitHub repository public only after the secret/path scan is clean:

   ```bash
   rg -n '/Users/gustavo|Traceback|AIza|gho_|ghp_|GOOGLE_API_KEY=.' \
     data/precomputed cartograph tests IMPLEMENTATION_PLAN.md README.md .env.example \
     --glob '!*.pyc'
   ```

   Expected matches are only README placeholders and test fixtures.

3. Prepare Devpost with:
   - GitHub repository URL.
   - Four-minute recording URL.
   - Draft copy from `docs/devpost_draft.md`.
   - Short description from `README.md`.
   - Technical explanation from `docs/cartograph_proposal.md`.
   - Verification summary from `IMPLEMENTATION_PLAN.md`.
