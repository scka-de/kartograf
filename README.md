# Kartograf

> *Plot the embeddings. See the gaps. Close them.*

**Kartograf** audits an AI agent's eval suite by mapping the *conceptual surface area*
the agent declares (its tool descriptions, system prompts, docstrings, eval inputs) into
a 2D semantic map, then finding the regions where the agent claims to do things its
existing tests do not exercise — and proposing BDD-style eval cases to close each gap.

It runs on a single Python agent repository. **No external corpus, no production traces,
no labeled dataset required** — the repo is the ground truth.

```bash
kartograf map --target ./path/to/agent --output map.html
open map.html
```

You see a 2D map: surfaces colored by kind (tool / prompt / docstring), eval cases
overlaid as diamonds, and red regions where surfaces cluster but no test reaches.
Hover any point for source. Click a gap region for a Gemini-generated BDD eval case
grounded in the surrounding surfaces.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[full,dev]"
cp .env.example .env
# put GOOGLE_API_KEY in .env (Gemini API enabled)
```

The `full` extra includes `google-genai`, `umap-learn`, `scipy`. Without `GOOGLE_API_KEY`
the pipeline falls back to a deterministic local embedder; coverage numbers will be
artificially high.

## Quickstart

### Audit one of the bundled ADK samples

```bash
kartograf map \
  --target data/adk_samples/python/agents/customer-service \
  --output docs/figures/kartograf_map_customer_service.html
open docs/figures/kartograf_map_customer_service.html
```

### Audit your own agent repo

```bash
kartograf map --target /path/to/your/agent --output map.html
```

Kartograf parses every `.py` file under `--target`, harvests:

- **Tool descriptions** — first paragraph of each public function's docstring
- **Tool args** — the `Args:` block of each tool docstring
- **Prompt sentences** — top-level string constants matching `*INSTRUCTION*` / `*PROMPT*` / `*SYSTEM*`
- **Public docstrings** — first paragraph of every public class and function

…and every `.evalset.json` / `.test.json` for eval inputs. If the surface count is
under 30, Kartograf refuses to score and asks you to enrich the agent's declared
contract first.

### Read the map

The interactive HTML page has the map on the left and a sidebar with:

- A coverage score (0–1)
- Gap regions ranked by integrated mass
- For each region: 5 nearest surfaces and a Gemini-generated BDD eval case ready to copy
  into your test suite

For a complete didactic walkthrough of what every element of the map means, see
[USER_GUIDE.md](USER_GUIDE.md).

## How the math works

Two density fields over the 2D UMAP projection:

```
ρ_S(x)  = surface density       (kernel density estimate of all declared surfaces)
ρ_E(x)  = eval density          (kernel density estimate of eval inputs)
g(x)    = max(0, ρ_S - α·ρ_E)   (the "gap field")
```

Connected components of `g(x) > τ` are gap regions, ranked by `∫ g dA`. For each region,
the k nearest surfaces in the original 768-dimensional embedding space become the
context for a Gemini call that produces one BDD-formatted eval case
(*Scenario / Given / When / Then*), validated against existing eval queries for
redundancy and against the surfaces for off-spec drift.

`α` defaults to `|S|/|E|` so a small eval suite is not penalized purely for being
outnumbered. Override with `--alpha`. `τ` is the 75th percentile of `g` on the surface
support.

The full proposal lives in [docs/proposal.md](docs/proposal.md).

## Architecture

```
cartograph/
├── core/           # shared math: embeddings (Gemini), 2D reducer
├── surfaces.py     # AST-based surface harvesting from a Python repo
├── evals.py        # eval input extraction (.evalset.json, .test.json)
├── density.py      # KDE + gap field + connected-component gap detection
├── generator.py    # Gemini-grounded BDD eval case synthesis with validation
├── pipeline.py     # end-to-end orchestration
├── report.py       # self-contained interactive HTML (Plotly)
└── cli.py          # kartograf version | map
```

## Development

```bash
pytest
ruff check .
mypy cartograph/core
```

Full verification (re-runs everything plus checks the hero artifacts are fresh):

```bash
scripts/verify.sh
```

The hero artifacts (`docs/figures/kartograf_map_*.html`) must exist and not be older
than any source file in `cartograph/`.

## Platform support

Linux and macOS. On macOS x86_64, the `full` extra falls back to a deterministic SVD
reducer instead of forcing a local llvmlite build for UMAP.

## License

Apache-2.0 — see [LICENSE](LICENSE).
