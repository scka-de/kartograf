# Cartograph Implementation Plan

Source of truth: `docs/cartograph_implementation_spec.md`. The spec wins over `docs/cartograph_proposal.md` when details disagree.

## Scope

Build Cartograph v1 as a Python package with:

- CLI commands: `cartograph audit`, `cartograph report`, `cartograph demo prepare`, `cartograph demo run`, `cartograph demo visualize`, `cartograph corpus list`, `cartograph version`.
- Core models, storage, embeddings, clustering, coverage, risk, validation, and ADK eval subprocess support.
- ADK-shaped root/mapper/auditor/generator agents with decision logging.
- Six runnable MCP adapter/provider modules.
- Demo precomputation, customer-service deep-dive, visualization, committed mock corpora, and `scripts/run_demo.sh`.
- README and tests that document current acceptance state.

## Build Checklist

| Step | Spec section | Artifact | Status | Notes |
|---|---:|---|---|---|
| 1 | 5 | Repo skeleton/tooling | Done | `pyproject.toml`, package dirs, tests, scripts, env/gitignore/pre-commit; local git repository initialized; `.env.example` documents Google ADK runtime modes and demo cache roots; `full` extra installs on macOS x86_64 by using the deterministic reducer fallback instead of forcing a local `llvmlite` build; runtime caches are gitignored. |
| 2 | 6 | Pydantic data models | Done | `cartograph/core/models.py`. |
| 3 | 7.5 | Storage | Done | SQLite + artifact paths in `cartograph/core/storage.py`. |
| 4 | 7.1 | Embeddings | Done | Gemini wrapper with deterministic local fallback for tests/demo without API key. |
| 5 | 7.2 | Clustering | Done | Uses UMAP/HDBSCAN if installed; deterministic sklearn/numpy fallback otherwise. |
| 6 | 7.3-7.4 | Coverage + risk | Done | Pure functions over models and 30d arrays. |
| 7 | 7.6 | ADK eval runner | Done | Subprocess wrapper, parser, timeout handling. |
| 8 | 10.1 | `corpus_reader_bitext` MCP | Done | Real HF path when available, cache, fixture fallback, optional MCP SDK stdio runner. |
| 9 | 10.4 | `eval_io_adk` MCP | Done | Read/write ADK evalset with unknown-field preservation. |
| 10 | 8.1 | Mapper agent | Done | ADK-compatible Python implementation with decision logging; ADK prompt constant matches spec. |
| 11 | 8.2 | Auditor agent | Done | Eval projection, assignment, coverage, gaps, redundancies; ADK prompt constant matches spec. |
| 12 | 8.3 | Generator agent | Done | Candidate generation plus redundancy/off-corpus validation; ADK prompt constant matches spec. |
| 13 | 10.6 | `coverage_state` MCP | Done | Provider functions backed by storage, optional MCP SDK stdio runner. |
| 14 | 9 | Root orchestrator | Done | Sequential autonomous workflow with loop limit; re-audits the merged generated evalset after generation; ADK prompt constant matches spec. |
| 15 | 11 | CLI | Done | Typer app with report/demo/corpus commands; `audit` streams decision rows as they are logged. |
| 16 | 10.2-10.3 | StackExchange/GitHub readers | Done | Live/cache/mock fallback ladder, optional MCP SDK stdio runner; bundled fallback fixtures now contain 30 rows each. |
| 17 | 10.5 | `eval_io_git` MCP | Done | YAML/JSONL round trip, optional MCP SDK stdio runner. |
| 18 | 12.1 | Demo prepare | Done | Produces fleet benchmark with real/mock corpus modes. |
| 19 | 12.2 | Live deep-dive | Done | Customer-service path emits required stdout fields and reports true before/after coverage. |
| 20 | 12.3 | Visualization | Done | Matplotlib PNG generation, placeholder fallback if matplotlib unavailable. |
| 21 | 13.1 | Acceptance script + README | Done | `scripts/run_demo.sh` verifies required files/strings. |
| 22 | 5 | Helper scripts | Done | Added `fetch_bitext.py`, `fetch_stackexchange.py`, `fetch_github_issues.py`, and `clone_adk_samples.py`. |

## Verification Checklist

| Requirement | Evidence | Status |
|---|---|---|
| `pip install -e .` succeeds | `.venv/bin/python -m pip install -e .` succeeded after network escalation for build/runtime dependencies. | Passed |
| `pip install -e ".[full,dev]"` succeeds | `.venv/bin/python -m pip install -e ".[full,dev]"` succeeded after making `umap-learn==0.5.5` conditional on non-macOS-x86_64 platforms; this environment uses the deterministic reducer fallback. | Passed |
| `cartograph version` prints version | `.venv/bin/cartograph version` printed `0.1.0`; `python -m cartograph.cli version` also printed `0.1.0`. | Passed |
| `cartograph corpus list` prints adapters | `.venv/bin/cartograph corpus list` printed `bitext`, `github`, and `stackexchange`. | Passed |
| `pytest` passes | `.venv/bin/python -m pytest` passed: 60 passed, 1 skipped, 61 collected. | Passed |
| `mypy cartograph/core` passes | `python -m mypy cartograph/core` passed with `strict = true` and missing-import overrides for optional `umap`, `hdbscan`, `joblib`. | Passed |
| `ruff check cartograph` passes | `.venv/bin/python -m ruff check cartograph tests scripts` passed. | Passed |
| `bash scripts/run_demo.sh` exits 0 | Passed for latest full-script audit `audit_6a42a655f8cc`; generated required JSON/PNG/reducer artifacts and stdout strings, including `adk_eval_blocker`. Customer-service coverage moved `0.67 -> 1.00`. | Passed |
| Mock fallback corpus sizes | `scripts/fetch_stackexchange.py --site money --limit 50` returned `{"count": 30, "mode": "mocked"}` before full extras were installed; `scripts/fetch_github_issues.py --owner psf --repo requests --limit 20` returned `{"count": 20, "mode": "mocked"}` from the 30-row fixture. | Passed |
| Live StackExchange corpus path | After full extras install, `.venv/bin/python scripts/fetch_stackexchange.py --site money --limit 50` returned `{"count": 50, "mode": "real"}`. | Passed |
| Live GitHub corpus path | After allowing unauthenticated public PyGithub access, `.venv/bin/python scripts/fetch_github_issues.py --owner psf --repo requests --limit 20` returned `{"count": 20, "mode": "real"}` without `GITHUB_TOKEN`. | Passed |
| CLI decision streaming | `tests/test_orchestrator_decisions.py::test_decision_observer_streams_logged_decisions` verifies a scoped observer receives each `log_decision` call while storage still persists the row. | Passed |
| MCP module docstrings | AST check confirmed docstrings exist in all six MCP server modules. | Passed |
| MCP modules start standalone | A subprocess smoke test started all six `python -m cartograph.mcp_servers.*` modules for 0.5s and terminated them cleanly after confirming they stayed alive. | Passed |
| MCP stdio client/server round trip | `tests/test_mcp_stdio_roundtrip.py` uses the real MCP SDK `stdio_client` and `ClientSession` to verify all six servers expose expected tool names, and to call `write_jsonl` / `read_jsonl` on `cartograph.mcp_servers.eval_io_git`. | Passed |
| Required precomputed files | `data/precomputed/fleet_benchmark.json`, `data/precomputed/customer_service_report.json`, and `data/precomputed/customer-service.evalset.json` exist and are non-empty. | Passed |
| Precomputed artifacts are publishable | `data/precomputed/customer_service_report.json` is ~16 KB after ADK output summarization, `customer-service_cartograph_generated.evalset.json` contains current-schema ADK cases and validates with `google.adk.evaluation.eval_set.EvalSet`; `rg` found no local paths or stack traces in `data/precomputed`. | Passed |
| `.env.example` documents runtime variables | `.env.example` lists `GOOGLE_API_KEY`, `GITHUB_TOKEN`, `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `MPLCONFIGDIR`, and `XDG_CACHE_HOME`. | Passed |
| Real ADK factory imports | `.venv/bin/python` with `google-adk==1.32.0` built root `cartograph` with sub-agents `mapper`, `auditor`, `generator`. | Passed |
| Real MCP SDK import | `.venv/bin/python` imported `mcp.server.fastmcp.FastMCP` and `run_stdio_server`. | Passed |
| Real Bitext corpus path | `.venv/bin/python` downloaded/cached HuggingFace Bitext and reported 26,872 examples with `mode: real`. | Passed |
| Real ADK sample checkout | `scripts/clone_adk_samples.py` cloned `google/adk-samples` into `data/adk_samples`; customer-service eval path is converted from the sample's `simple.test.json`. | Passed |
| Real ADK eval reaches model call | Escalated `CARTOGRAPH_RUN_E2E=1 GOOGLE_GENAI_USE_VERTEXAI=1 .venv/bin/python -m pytest -m e2e` invoked `.venv/bin/adk eval` against the cloned customer-service module and converted evalset using Vertex AI project `kartograf-495419`; latest audit `audit_c390577a669a` completed with coverage `1.00`, before pass/fail `0/2`, and after pass/fail `0/14`. | Passed |
| Opt-in live e2e smoke test | `tests/test_e2e_smoke.py` now runs `run_customer_service_deep_dive(limit=30)` when `CARTOGRAPH_RUN_E2E=1` and Google runtime credentials are configured, then asserts coverage threshold and non-null before/after pass rates. Escalated Vertex run passed: 1 passed, 60 deselected in 62.91s. | Passed |
| Repo-wide code review hardening | Review fixes added `.env` loading to the ADK subprocess env, current-schema ADK generated evalset output, cumulative generated-case persistence across rounds, and an explicit empty-corpus failure. Targeted regression tests and full gates pass. | Passed |
| Final pause audit | Re-ran `.venv/bin/python -m pytest`, `.venv/bin/ruff check .`, `.venv/bin/mypy cartograph/core`, `.venv/bin/cartograph version`, `.venv/bin/cartograph corpus list`, a precomputed artifact/schema inspection, and `bash scripts/run_demo.sh`. The local acceptance script passed for `audit_8755b0e3b3c0`; because sandboxed ADK eval degraded to `adk_eval_blocker`, its precomputed artifact mutations were restored to the previously committed Vertex-verified audit `audit_c390577a669a`. | Passed with noted runtime constraint |
| Submission checklist | `docs/submission_checklist.md` now gives the live preflight commands, four-minute recording arc from proposal section 9, fallback/Vertex artifact warning, public GitHub preflight scan, and Devpost payload list. README links to it from the Demo section. | Prepared; external execution still required |
| Devpost draft | `docs/devpost_draft.md` contains project name, tagline, links placeholders, inspiration, what it does, build details, demo arc, challenges, accomplishments, learnings, next steps, built-with list, and verification summary. | Prepared; URL/video placeholders still required |
| Non-mutating pre-submission check | `scripts/pre_submission_check.sh` runs tests, ruff, mypy, CLI smoke checks, validates committed precomputed artifacts and generated ADK evalset schema, checks non-null pass rates, and performs the publishability scan without rewriting demo artifacts. | Added; run before public release |
| GitHub Actions CI | `.github/workflows/ci.yml` installs `.[full,dev]` on Python 3.11 and runs `scripts/pre_submission_check.sh` on pushes to `main` and pull requests. | Added; remote run pending after push |

## Completion Audit

Concrete deliverables requested by the prompt:

- Read documents under `docs/`: done for `cartograph_implementation_spec.md` and `cartograph_proposal.md`; SVGs were listed as supporting diagrams.
- Create a new markdown plan based on `cartograph_implementation_spec.md`: done in this file.
- Implement the spec: repo/runtime implementation is complete for v1 acceptance, including live Vertex ADK eval verification; final submission readiness still depends on external artifacts.
- Keep the plan updated if paused: this file reflects current status and remaining gaps.

### Prompt-to-Artifact Checklist

| Objective / spec requirement | Artifact or command inspected | Evidence | Status |
|---|---|---|---|
| Read documents under `/docs` | `docs/cartograph_implementation_spec.md`, `docs/cartograph_proposal.md`, listed SVGs | Spec and proposal were read; implementation follows spec where proposal differs. | Done |
| Create a new markdown plan based on the implementation spec | `IMPLEMENTATION_PLAN.md` | This plan tracks build order, verification, completion audit, constraints, and blockers. | Done |
| Local git repository exists | `.git/`, `git status --short --untracked-files=all`, `git log --oneline -1` | `git init` succeeded; publishable surface contains source/docs/tests/scripts and `data/precomputed`, while runtime caches are ignored; initial snapshot committed as `78739a3 Initial Cartograph implementation`. | Done |
| Python package `cartograph` installs | `pyproject.toml`, `.venv/bin/python -m pip install -e .`, `.venv/bin/python -m pip install -e ".[full,dev]"` | Base and full/dev installs passed in this environment after macOS x86_64 UMAP marker fix. | Done |
| CLI commands exist | `cartograph/cli.py`, `.venv/bin/cartograph version`, `.venv/bin/cartograph corpus list`, `scripts/run_demo.sh` | `version`, `audit`, `report`, `demo prepare`, `demo run`, `demo visualize`, `corpus list` implemented; version/corpus commands verified. | Done |
| CLI `audit` streams decision rows live | `cartograph/agents/common.py`, `cartograph/cli.py`, `tests/test_orchestrator_decisions.py` | Scoped decision observer emits rows as `log_decision` persists them. | Done |
| Core models match stable schema | `cartograph/core/models.py`, tests | Pydantic v2 models implemented and used across storage/agents/tests. | Done |
| Storage persists SQLite plus audit artifacts | `cartograph/core/storage.py`, `data/audits/*`, `tests/test_core_storage.py` | Reports load from DB and artifact files; reducers/embeddings/visualizations are created by demo. | Done |
| Embeddings wrapper and cache | `cartograph/core/embeddings.py`, `data/embeddings/*`, demo output | Gemini path is used when `GOOGLE_API_KEY` exists; deterministic fallback keeps local demo/tests runnable. | Done |
| Clustering, reducers, and 30d convention | `cartograph/core/clustering.py`, `cartograph/core/math.py`, `tests/test_core_clustering.py` | 30d reducer and 2d visual reducer are persisted; tests cover reducer round trip and dimensions. | Done |
| Coverage and risk scoring | `cartograph/core/coverage.py`, `cartograph/core/risk.py`, tests | Pure functions verified with unit tests. | Done |
| ADK eval subprocess wrapper | `cartograph/core/adk_eval.py`, `tests/test_core_adk_eval.py`, live demo reports | Parser/env/timeout behavior tested; live command reaches Vertex model runtime and parses ADK `Tests passed` / `Tests failed` summaries. | Done |
| ADK-shaped root and sub-agents | `cartograph/agents/*.py`, `cartograph/agents/adk_factory.py`, `tests/test_agent_adk_factory.py` | Real `google-adk==1.32.0` factory builds root `cartograph` with mapper/auditor/generator sub-agents. | Done |
| Root orchestrator workflow and loop limit | `cartograph/agents/root.py`, `tests/test_orchestrator_decisions.py`, latest demo output | Re-audits merged generated evalset; customer-service terminates after one generation round at coverage `1.00`; loop limit implemented. | Done |
| Generator validates redundancy and off-corpus cases | `cartograph/agents/generator.py`, `cartograph/core/validation.py`, tests, demo output | Negative-control duplicate is rejected; accepted cases cover target region; off-corpus path covered by tests. | Done |
| Six MCP modules are runnable | `cartograph/mcp_servers/*.py`, subprocess smoke start | All six `python -m cartograph.mcp_servers.*` modules start under installed MCP SDK. | Done |
| MCP stdio client/server behavior | `tests/test_mcp_stdio_roundtrip.py` | Real MCP SDK lists expected tools for all six stdio servers and calls `write_jsonl` / `read_jsonl` over stdio against `eval_io_git`. | Done |
| Bitext corpus reader real path | `cartograph/mcp_servers/corpus_reader_bitext.py`, `data/corpora/bitext.json` | Real HuggingFace Bitext cache has 26,872 examples and customer-service fleet entry is `real`. | Done |
| StackExchange corpus reader real path | `scripts/fetch_stackexchange.py --site money --limit 50` | Returned 50 real rows after full extras install. | Done |
| GitHub corpus reader real path | `scripts/fetch_github_issues.py --owner psf --repo requests --limit 20` | Returned 20 real rows unauthenticated. | Done |
| Eval I/O ADK round trip | `cartograph/mcp_servers/eval_io_adk.py`, `tests/test_mcp_eval_io_adk.py`, `data/precomputed/customer-service.evalset.json` | Supports current adk-samples `.test.json`, old list format, and evalset object format. | Done |
| Eval I/O Git round trip | `cartograph/mcp_servers/eval_io_git.py`, tests | JSONL round trip and real MCP stdio round trip pass. | Done |
| coverage_state provider | `cartograph/mcp_servers/coverage_state.py`, `tests/test_mcp_coverage_state.py` | Lists audits, gaps, generated cases; full report provider implemented. | Done |
| Fleet benchmark for 5 agents | `data/precomputed/fleet_benchmark.json`, `bash scripts/run_demo.sh` | Latest acceptance run shows 5 entries; customer-service/software-bug/financial real, travel/data-science mocked fallback. | Done under fallback acceptance |
| Live customer-service deep dive | `cartograph/demo/deep_dive.py`, `data/precomputed/customer_service_report.json`, `scripts/run_demo.sh` | Prints coverage before/after, generated cases, decision log, pass-rate lines, blocker line, and rejection event; latest precomputed report has non-null before/after pass-rate values and summarizes raw ADK output for publication. | Done |
| Visualization PNGs | `cartograph/demo/visualize.py`, `data/audits/<audit_id>/viz/*.png` | `regions.png`, `coverage.png`, `before_after.png` generated by acceptance script. | Done |
| Final acceptance script | `scripts/run_demo.sh` | Latest local verification exited 0 for `audit_8755b0e3b3c0`; generated required visualizations and stdout contract. Precomputed report remains restored to the Vertex-verified `audit_c390577a669a` because the local sandboxed run could not complete live ADK eval. | Done |
| Non-mutating pre-submission check | `scripts/pre_submission_check.sh` | Validates the committed artifact state without running a new audit, so it is safer than `run_demo.sh` immediately before making the repository public. | Done |
| CI workflow | `.github/workflows/ci.yml` | Runs the non-mutating pre-submission check in GitHub Actions for `main` and pull requests. | Done locally; remote status pending |
| README smoke-demo docs | `README.md` | Install, env vars, quickstart, demo command, e2e command, proposal/spec links, runtime notes documented. | Done |
| Tests and quality gates | `.venv/bin/python -m pytest`, `ruff`, `mypy cartograph/core` | Latest full suite: 60 passed, 1 skipped, 61 collected; ruff and mypy pass. | Done |
| Live e2e with real pass rates | `tests/test_e2e_smoke.py`, `CARTOGRAPH_RUN_E2E=1 ... pytest -m e2e` | Escalated Vertex run passed with audit `audit_c390577a669a`, coverage `1.00`, non-null before pass rate `0.0`, and non-null after pass rate `0.0`. | Done |
| Submission recording | External artifact | Not present in workspace; requires human recording. | Blocked externally |
| GitHub repository and devpost link | External publishing artifacts | Private interim repo exists at `https://github.com/scka-de/kartograf` and `main` tracks `origin/main`. `docs/submission_checklist.md` documents the final public-repo scan; `docs/devpost_draft.md` provides the Devpost copy. Final submission still requires explicit approval to make the repo public, a recording URL, and the final Devpost URL. | Partially done |

Current high-confidence evidence:

- Package skeleton, CLI, agents, MCP boundary modules, core modules, demo modules, fixtures, README, tests, and acceptance script exist.
- Local git repository exists; initial implementation snapshot committed as `78739a3 Initial Cartograph implementation`; private remote `origin` is `https://github.com/scka-de/kartograf.git`.
- Final script `bash scripts/run_demo.sh` exits 0 and checks the required files/stdout contract with `customer-service` in `corpus_mode == "real"`; the most recent sandboxed local run passed for `audit_8755b0e3b3c0`.
- The customer-service deep dive now invokes Generator once, re-audits the merged original-plus-generated evalset, and terminates at coverage `1.00`.
- Repo-wide review fixes are covered by regressions: ADK `.env` loading, current ADK evalset serialization for generated cases, cumulative generated-case storage, and explicit failure on empty adapter corpora.
- The `cartograph audit` CLI path now streams decision rows as they are written through a scoped observer hook.
- Local quality gates pass: ruff, mypy core strict, pytest.
- Root files, MCP docstrings, README runtime/e2e notes, `.env.example`, and required precomputed demo artifacts were checked directly.

Remaining gaps before the full spec can honestly be called complete:

- ADK agent factory exists in `cartograph/agents/adk_factory.py` and was verified against installed `google-adk==1.32.0`; full ADK runner pass-rate execution is now verified through the live Vertex e2e smoke test.
- MCP modules instantiate an MCP SDK stdio server when `mcp.server.fastmcp` is installed; real stdio tool discovery is covered for all six modules and a call round trip is covered for `eval_io_git`.
- The suite exceeds the spec target of 30 collected tests and includes `test_agent_generator.py` plus a real opt-in `test_e2e_smoke.py`; the opt-in e2e passed with Vertex credentials and remains skipped by default.
- Fleet fallback behavior is now honest: `customer-service` uses the real cached HuggingFace Bitext path; StackExchange `money` and GitHub `psf/requests` live paths were verified after full extras install; bundled fallback fixtures now satisfy the 30-item spec assumption.
- Real `adk eval` is now wired against the checked-out customer-service sample and converted evalset, and the Vertex path has been exercised successfully. Current committed customer-service ADK pass rates are non-null but low: before `0/2`, after `0/14`; this is enough for the pass-rate comparison contract but not an agent-quality win.
- Submission-readiness items remain out of band: 4-minute recording, making the GitHub repo public, and devpost link. The GitHub repo is intentionally private for now per the user's earlier direction; making it public needs explicit confirmation.

## Known Constraints

- This workspace started without `.git`; local git metadata, commits, and private remote `origin` now exist.
- External APIs and Gemini may be unavailable in the sandbox. The implementation includes deterministic local fallbacks so tests and demo scaffolding remain runnable without credentials.
- When `scripts/run_demo.sh` is run without external Google/Vertex access, the script can still pass the fallback acceptance contract but may overwrite `data/precomputed/customer_service_report.json` with `pass_rate_before: None`, `pass_rate_after: None`, and `adk_eval_blocker`. Restore or rerun the live Vertex e2e before publishing precomputed artifacts.
- On macOS x86_64, `umap-learn==0.5.5` may require building `llvmlite` from source with CMake. The `full` extra skips UMAP on that platform and uses Cartograph's persisted projection reducer fallback; Linux and other platforms can still install the pinned UMAP package.
- Full submission readiness still requires a human-recorded 4-minute demo and public GitHub/devpost steps.
