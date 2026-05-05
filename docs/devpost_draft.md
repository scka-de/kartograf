# Cartograph Devpost Draft

Use this as the starting payload for Devpost. Replace the placeholders before
submission.

## Project Name

Cartograph

## Tagline

Semantic coverage for agent eval suites.

## Links

- GitHub: `TODO: make https://github.com/scka-de/kartograf public and paste URL`
- Demo video: `TODO: paste four-minute recording URL`
- Live demo command: `bash scripts/run_demo.sh`

## Inspiration

Agent teams often treat eval pass rate as a release signal, but pass rate only
answers whether the agent succeeds on the cases the team remembered to test.
It does not answer whether those tests represent the real semantic world the
agent will face. Cartograph was built around that gap: before asking whether an
agent is correct everywhere, first ask whether its tests are looking in the
right places.

## What It Does

Cartograph is an autonomous ADK multi-agent system that audits another agent's
eval suite against a real-world corpus.

It:

- Loads a real corpus through MCP adapters.
- Embeds and clusters the corpus into semantic regions.
- Projects existing eval cases into the same region map.
- Computes semantic coverage and risk-ranked gaps.
- Generates new corpus-grounded eval cases for uncovered regions.
- Rejects redundant and off-corpus generated cases.
- Re-runs ADK eval before and after generation.
- Exposes coverage state through an MCP provider.

The output is a report that shows what the current eval suite covers, what it
misses, and which generated cases would make the test suite more honest.

## How We Built It

Cartograph is implemented as a Python package with:

- A root ADK agent plus Mapper, Auditor, and Generator sub-agents.
- Six MCP servers:
  - `corpus_reader_bitext`
  - `corpus_reader_stackexchange`
  - `corpus_reader_github`
  - `eval_io_adk`
  - `eval_io_git`
  - `coverage_state`
- Gemini embeddings with deterministic local fallbacks.
- 30-dimensional region-space projection for clustering, risk, and assignment.
- SQLite metadata plus persisted embedding/reducer artifacts.
- A Typer CLI for audit, report, corpus, and demo flows.
- A demo pipeline across five `google/adk-samples` agents.

The demo uses ADK samples because they are educational reference agents with
illustrative eval suites. Cartograph measures the distance between an
illustrative eval suite and a production-representative input distribution; the
same gap exists in real agent deployments.

## Demo Arc

The four-minute recording follows this arc:

1. Frame the problem: high eval pass rate does not imply representative tests.
2. Run the five-agent fleet benchmark.
3. Deep-dive on `customer-service` with the Bitext support corpus.
4. Show the decision log: ingest, map, audit, generate, re-audit.
5. Show generated cases and rejected controls.
6. Compare before/after ADK pass rates and semantic coverage.
7. Close with the thesis: the agent did not get worse; the test got more honest.

## Challenges

- ADK eval output and evalset schemas are evolving, so Cartograph keeps the ADK
  CLI as the runtime contract and validates generated evalsets against the real
  ADK `EvalSet` model.
- External corpora and APIs can fail during demos. The fleet benchmark records
  `corpus_mode` explicitly and supports fallback fixtures while requiring the
  `customer-service` Bitext path to remain real.
- Generated eval cases must be grounded. Cartograph validates candidates against
  corpus exemplars and existing evals instead of blindly appending LLM output.

## Accomplishments

- Built a working autonomous ADK multi-agent audit loop.
- Shipped MCP adapters for corpora, eval I/O, and coverage-state consumption.
- Verified live Vertex ADK eval execution with non-null before/after pass rates.
- Added deterministic tests for coverage, clustering, risk, validation, ADK eval
  parsing, MCP stdio round trips, and orchestration decisions.
- Produced committed precomputed demo artifacts for reproducible review.

## What We Learned

The useful signal is not "more evals." It is whether each new eval is grounded
in a real uncovered semantic region. Coverage becomes more actionable when it is
linked to exemplars, risk ranking, and generated test cases that can be reviewed
or committed.

## What's Next

- Add adapters for LangSmith, OpenAI evals, and custom YAML eval suites.
- Add a review UI for accepting or rejecting generated cases.
- Track semantic coverage drift over time as production corpora change.
- Integrate with CI so pull requests can fail when eval suites lose coverage.
- Extend from L1 semantic coverage into behavioral and trajectory coverage.

## Built With

- Google ADK
- Gemini / Vertex AI
- MCP
- Python 3.11
- Typer
- Pydantic v2
- SQLite
- NumPy
- scikit-learn
- Matplotlib

## Verification Summary

Latest committed verification summary:

- Unit/integration suite: `60 passed, 1 skipped`
- Ruff: passed
- Mypy for `cartograph/core`: passed
- Live Vertex e2e smoke: passed
- Latest Vertex-verified audit: `audit_c390577a669a`
- Customer-service semantic coverage: `1.0`
- ADK pass rates before/after: `0.0 / 0.0`

The ADK pass rates are intentionally not framed as a quality win. They prove the
before/after pass-rate comparison is wired. The product thesis is semantic
coverage: making the test suite more representative of the world the agent will
face.
