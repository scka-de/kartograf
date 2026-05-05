# Cartograph

Cartograph audits whether an agent eval suite represents the semantic world the agent actually faces. It maps a real corpus, projects existing evals into that map, reports semantic coverage gaps, and proposes corpus-grounded eval cases.

The builder-facing source of truth is [docs/cartograph_implementation_spec.md](docs/cartograph_implementation_spec.md). The judge-facing proposal is [docs/cartograph_proposal.md](docs/cartograph_proposal.md).

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For live external integrations:

```bash
pip install -e ".[full,dev]"
cp .env.example .env
```

Set `GOOGLE_API_KEY` for Gemini embeddings/generation. Set `GITHUB_TOKEN` only for private repositories or higher GitHub API limits; public demo issue fetches work unauthenticated. Without keys, Cartograph uses deterministic local fallbacks so the smoke demo remains runnable.

For real `adk eval` pass-rate verification, configure one of the Google runtime paths:

- Gemini Developer API: set `GOOGLE_API_KEY` and `GOOGLE_GENAI_USE_VERTEXAI=0`.
- Vertex AI: set `GOOGLE_GENAI_USE_VERTEXAI=1`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION`, then enable the Agent Platform API for that project. `GOOGLE_CLOUD_PROJECT` must be the project ID, not the display name.

## Quickstart

```bash
cartograph version
cartograph audit --target python/agents/customer-service --corpus bitext --eval-path data/precomputed/customer-service.evalset.json --corpus-limit 30
cartograph report --latest
```

## Demo

Warm-cache smoke demo:

```bash
bash scripts/run_demo.sh
```

Manual steps:

```bash
cartograph demo prepare
cartograph demo run customer-service
cartograph demo visualize <audit_id>
```

The full cold prepare path can take up to 30 minutes with live HuggingFace, StackExchange, GitHub, and Gemini calls. The fallback path uses committed fixtures for non-customer-service corpora and still emits `corpus_mode` so the recording can be narrated honestly.

## Development

```bash
pytest
ruff check cartograph
mypy cartograph/core
```

Live pre-submission smoke test:

```bash
CARTOGRAPH_RUN_E2E=1 GOOGLE_API_KEY=... GOOGLE_GENAI_USE_VERTEXAI=0 pytest -m e2e
```

## ADK and MCP Runtime Notes

The package can run offline for smoke tests. When `google-adk` is installed, `cartograph.agents.adk_factory.build_cartograph_adk_agent()` constructs a real ADK `LlmAgent` root with Mapper, Auditor, and Generator sub-agents. When `mcp` is installed, each `python -m cartograph.mcp_servers.*` module starts a stdio MCP server; otherwise it prints the available tool list for local inspection.

Linux and macOS are supported for v1.

On macOS x86_64, the `full` extra uses Cartograph's deterministic reducer fallback instead of forcing a local `llvmlite` build for UMAP.
