# Kartograf

> *Plot the embeddings. See the gaps. Close them.*

---

## 1. What Kartograf Is

Kartograf is an autonomous agent that audits another agent's eval suite by
**embedding every surface the agent exposes**, **plotting the embeddings in a 2D semantic
map**, **finding the regions where the agent claims to do things its tests do not
exercise**, and **proposing eval cases grounded in the real surfaces around each
gap** — using only the target repo as input.

> **Kartograf turns the embedding plot itself into the discovery instrument. The map shows
> what the agent claims to do. The empty regions on the map are what nobody is testing.**

---

## 2. The Problem

Agents pass evals and break in production because no one measures whether the eval suite
exercises everything the agent claims to do.

A team writes 15–30 hand-crafted eval cases, hits 90%+ pass rate, and ships. Two weeks
later, production users hit behaviors the agent declared in its tool schemas, prompts, and
docstrings — but no eval ever exercised. Refunds adjacent to cancellations. Multi-tool
escalations. Privacy requests inside support flows. The agent had the surface area; the
eval suite had the imagination.

Existing tools either measure pass rate against the suite the team wrote (LangSmith,
Braintrust) or apply runtime guardrails after the fact (Patronus, Galileo). None of them
ask: **does the eval suite exercise the conceptual surface the agent itself declares?**

That is the gap Kartograf fills — and it answers it without the user bringing any
external corpus, dataset, or production trace. The repo contains everything the math
needs.

---

## 3. The Insight

Every agent repo already contains a structured description of itself, written by the
developer, in natural language:

| Surface | Source |
|---|---|
| **Tool descriptions** | Tool schema strings the dev wrote for the LLM |
| **System prompts** | The agent's self-description |
| **Eval case inputs** | What the test suite tries |
| **Docstrings** | Public function intent |
| **Example queries** | Whatever the README/docs put forward |

These surfaces describe the **conceptual surface area** of the agent — the world it claims
to handle. They are all natural language. They embed cleanly. And they are all in the repo
already.

Kartograf's central observation:

> If you embed every surface and every eval case into the same space, plot them in 2D, and
> overlay the eval-case density on the surface density, **the gaps in the plot are the gaps
> in the test suite**. You can literally see them.

The repo's own declared surfaces are the ground truth. The math operates on what is
already there — no external corpus, no labeled dataset, no production traces required.

---

## 4. Why an Agent, Not a Pipeline

Kartograf is autonomous. The orchestrator decides which surfaces to extract from the repo,
picks the bandwidth for density estimation based on point spread, ranks gap regions by
integrated gap density, decides how many candidates to retrieve from the neighborhood of
each gap, validates generated cases, and writes new evals back through MCP. Every step is
a runtime decision, logged for reproducibility.

This matters because the right thresholds depend on the repo. A 50-surface agent and a
500-surface agent both produce useful maps, but with different bandwidths, different
top-K values, different rejection cutoffs. A static pipeline would either over-fit to one
repo size or strand half its inputs.

---

## 5. How Coverage Is Measured

The full math runs in two spaces: a high-dimensional embedding space for retrieval, and a
2D projection for density estimation and visualization.

### 5.1 Surfaces and evals

Let:
- $S = \{s_1, \ldots, s_n\}$ = set of **surface texts** extracted from the repo (tool
  descriptions, system prompt sentences, docstrings, example queries)
- $E = \{e_1, \ldots, e_m\}$ = set of **eval case inputs** extracted from the existing
  test suite

Both are natural language. Both are already in the repo.

### 5.2 Embedding and projection

Each text $t$ is embedded with Gemini `text-embedding-004`:

$$\mathbf{v}(t) \in \mathbb{R}^{768}$$

UMAP fits a 2D projection over $S \cup E$:

$$\mathbf{u}(t) \in \mathbb{R}^2$$

The 2D space is where density estimation runs. The 768D space is where retrieval runs.
Both are persisted per audit.

### 5.3 Density fields

Gaussian kernel density estimation gives two scalar fields over the 2D plane:

$$\rho_S(\mathbf{x}) = \frac{1}{n} \sum_{s \in S} K_h(\mathbf{x} - \mathbf{u}(s)), \quad
\rho_E(\mathbf{x}) = \frac{1}{m} \sum_{e \in E} K_h(\mathbf{x} - \mathbf{u}(e))$$

with bandwidth $h$ chosen by Scott's rule on the combined point cloud.

### 5.4 The gap field

$$g(\mathbf{x}) = \rho_S(\mathbf{x}) - \alpha \cdot \rho_E(\mathbf{x})$$

Where $g(\mathbf{x}) > \tau$, surfaces are dense and evals are sparse — the agent claims
something there and no test exercises it. $\alpha$ defaults to $|S| / |E|$ (mass-balanced)
so a single eval is not penalized for being outnumbered, and $\tau$ defaults to the 75th
percentile of $g$ on the surface support.

**On $\alpha$.** Each density field integrates to 1 by construction. With $\alpha = 1$, an
eval suite of 18 cases must compete pointwise against a surface set of 148 — the eval
field is structurally lower in absolute density, $g$ is positive nearly everywhere, and
the gap signal collapses into "everything is uncovered." Mass-balancing rescales $\rho_E$
so the **integrated** test mass equals the integrated surface mass; what survives in $g$ is
spatial mismatch, not cardinality mismatch. We log $\alpha$ per audit and expose it as a
CLI override (`--alpha 1.0` recovers raw-density mode for users who want it). The
sensitivity analysis appendix in the demo notebook plots gap-region count vs $\alpha$ on
the customer-service benchmark.

### 5.5 Gap regions

Connected components of $\{\mathbf{x} : g(\mathbf{x}) > \tau\}$ are the **gap regions**
$R_1, \ldots, R_k$. Each region is ranked by:

$$\text{score}(R_i) = \int_{R_i} g(\mathbf{x}) \, d\mathbf{x}$$

— the integrated gap mass. A small intense hole and a large mild hole are both legible;
the integral compares them honestly.

### 5.6 Coverage score

$$\text{coverage} = 1 - \frac{\int g(\mathbf{x}) \, d\mathbf{x}}{\int \rho_S(\mathbf{x}) \, d\mathbf{x}}$$

A single number from 0 to 1: the fraction of declared surface mass that has eval mass
nearby. The headline answer to *"is this eval suite exercising what the agent claims?"*

### 5.7 From gap region to text

For each gap region $R$, take the centroid $\mathbf{c}_R = \arg\max_{\mathbf{x} \in R}
g(\mathbf{x})$. Retrieve the **k nearest surfaces** in the original 768D space:

$$\text{nbrs}(R) = \text{top-}k_{s \in S} \, \|\mathbf{v}(s) - \mathbf{v}(\mathbf{c}_R^\uparrow)\|$$

where $\mathbf{c}_R^\uparrow$ is the high-dimensional pre-image — defined as the nearest
real surface point in 2D mapped back to its 768D vector. **No invertible decoder is
claimed; the "text of the gap" is the textual neighborhood that bounds it.**

This is the precise mathematical operation that lets Kartograf go from "there is a hole at
(5.2, -1.3)" to "here are 5 real surfaces from the repo that surround this hole."

### 5.8 What this metric does and does not claim

- It measures whether eval cases are densely co-located with declared surfaces in
  semantic space. It does **not** certify the agent reasons correctly inside any region.
- It treats every declared surface as equally significant. A privacy clause and a small
  helper docstring weigh the same per token. Per-kind weights are on the roadmap.
- The score is meaningful only relative to what the developer actually declared. An agent
  with thin tool descriptions and no docstrings produces a small map and an easy 100%.
  Kartograf reports the surface count alongside the score so this is never ambiguous.

---

## 6. How Kartograf Works

Kartograf is a multi-agent ADK system: a root orchestrator, three specialized sub-agents,
one persistent state, one MCP boundary.

### 6.1 The three sub-agents

**Mapper.** Walks the target repo. Extracts every declared surface — parses tool schemas,
splits system prompts into sentences, pulls docstrings of public callables via AST,
collects example queries from docs. Embeds each surface with Gemini. Fits UMAP. Persists
the 768D vectors, the 2D projection, the reducer.

**Auditor.** Loads the existing eval suite, embeds eval inputs in the same space,
projects through the persisted UMAP. Computes $\rho_S$, $\rho_E$, $g$. Identifies
connected gap regions, ranks them by integrated gap mass, computes the coverage score.

**Generator.** For each top-ranked gap region, retrieves the $k$ nearest surface texts
in 768D, sends them to Gemini with the agent's system prompt as additional context, asks
for a single eval case that exercises the conceptual region those surfaces bound.
Validates each candidate against two filters: redundancy with existing evals (cosine
distance to nearest eval > $\delta_{\text{redund}}$) and off-spec drift (cosine distance
to nearest surface < $\delta_{\text{drift}}$). Surfaces accepted candidates with full
provenance.

### 6.2 The orchestrator

LLM-driven (Gemini 2.5 Pro). Reads the current state, decides:

- Which surface kinds to extract (the dev may opt out of any kind)
- Bandwidth $h$ for KDE if Scott's rule produces degenerate fields
- How many gap regions to target with Generator (default: top 3 by integrated mass)
- How many candidates to retrieve per region (default: 5)
- When to terminate (coverage above target, or all top regions resolved, or budget hit)

Every decision is logged with inputs, outputs, and a one-line reason. The decision log is
the artifact that makes Kartograf an agent rather than a pipeline.

### 6.3 Sample orchestrator log

The numbers in the log below are illustrative — drawn from the design budget for a
customer-service-sized agent. The committed hero artifacts (§9.1) carry real values from
end-to-end runs.

```
step 1  scan_repo       → 47 tool descriptions, 12 prompt sentences,
                          83 docstrings, 6 example queries
                          surface count: 148

step 2  invoke_mapper   → 148 surfaces embedded, projected to 2D
                          surface density field: ready

step 3  load_evals      → 18 eval cases from customer-service.evalset.json
                          eval density field: ready

step 4  invoke_auditor  → coverage = 0.43
                          4 gap regions detected, ranked by integrated gap mass:
                            R1: refund-near-cancellation       (mass 0.094)
                            R2: multi-tool escalation           (mass 0.067)
                            R3: privacy-inside-support          (mass 0.041)
                            R4: billing-dispute-with-evidence   (mass 0.022)

step 5  decide          → coverage 0.43 < threshold 0.75
                          target top 3 regions, 5 candidates each

step 6  invoke_generator → 15 candidates produced
                           3 rejected: redundant with existing evals
                           1 rejected: off-spec drift > 0.42
                           11 accepted, each grounded in 4–6 surface texts

step 7  invoke_auditor   → coverage = 0.61

step 8  decide          → R4 unresolved, but mass 0.022 below
                          mass-budget threshold of 0.025
                          terminate, surface report
```

The orchestrator's reasoning is the audit's accountability surface.

---

## 7. How MCP Fits

Kartograf plays two MCP roles.

**Kartograf as MCP client** — consumes:

- `repo_reader` — exposes the target repo's surfaces (tool schemas, prompts,
  docstrings, examples) as a queryable source
- `eval_io` — reads existing eval suites and writes new ones

**Kartograf as MCP provider** — exposes:

- `coverage_state` — the canonical Kartograf output: surface map, eval map,
  gap regions, ranked proposals

The provider role earns its place by being consumed end-to-end: a CI gate
(`scripts/ci_coverage_gate.py`) reads `coverage_state` over MCP, fails the build if the
score regressed against the repo's baseline, and posts the top-3 gap regions as a PR
comment.

### Reference MCP servers

| Server | Role | Purpose |
|---|---|---|
| `repo_reader_python` | Client adapter | AST-based extraction of tool/prompt/docstring surfaces from any Python ADK agent |
| `repo_reader_adk` | Client adapter | ADK-specific extraction: `LlmAgent`, `tools=[...]`, `instruction=...` patterns |
| `eval_io_adk` | Client adapter | Reads and writes ADK `.evalset.json` |
| `eval_io_pytest` | Client adapter | Reads `pytest.parametrize` cases as eval inputs |
| `eval_io_yaml` | Client adapter | Generic YAML eval suite I/O |
| `coverage_state` | Provider | Canonical Kartograf output with the full map and gap report |

---

## 8. What Kartograf Outputs

Every audit produces three artifacts. The first one is the centerpiece.

**1. The 2D semantic map.** An interactive HTML page (single self-contained file). Every
surface plotted as one color, every eval case as another, gap regions shaded by integrated
mass. Hover any point: see its source text. Click a gap region: see its top-k bounding
surfaces and the proposed eval case. **This is the product. It is the demo. It is what a
user looks at to decide whether to merge the proposed tests.**

**2. The coverage report.** A JSON document with the coverage score, per-region gap
mass, surface counts by kind, accepted/rejected candidate counts, full decision log.
Machine-readable, CI-ready.

**3. Proposed eval cases.** New test cases generated by the Generator, formatted in the
target framework's eval format (ADK `.evalset.json`, generic YAML, JSONL). Each case
includes provenance: which surfaces it was grounded in, which gap region it covers, why
the validation step accepted it. Surfaced for developer review, never auto-committed.

---

## 9. The Map

The map is the product.

**Reference target:** `python/agents/customer-service` from `google/adk-samples`.
**Eval suite under audit:** the agent's shipped `customer-service.evalset.json`.

### 9.1 Hero artifacts

The repo ships two reference maps, both generated end-to-end by `kartograf map` and
verified by `scripts/verify.sh` (fails if either is missing or older than any source in
`cartograph/`):

| File | Target | Coverage | What it shows |
|---|---|---:|---|
| `docs/figures/kartograf_map_customer_service.html` | `customer-service` | 0.907 | Healthy case: 91 surfaces, 12 evals, 2 narrow gaps. Kartograf finds gaps even on a comparatively well-tested agent. |
| `docs/figures/kartograf_map_financial_advisor.html` | `financial-advisor` | 0.372 | Severe case: 246 surfaces, 1 eval. The map is mostly orange and purple with a single dominant red region around the `trading_analyst` subagent — visual story readable in two seconds. |

Both are generated with Gemini `gemini-embedding-001` (output_dimensionality=768).

The figure shows the 2D map with:

- surface points (one color, one marker shape)
- eval points (different color, different marker)
- gap regions shaded by integrated $g$ mass, top-3 labeled with their bounding surfaces

Schematic:

```
                              ▲ UMAP-2
                              │
              ●●●●            │
            ●●● ▲▲▲           │      ● = surface (tool / prompt / docstring)
            ●●  ▲▲▲           │      ▲ = eval case
            ●●●●▲             │      ░ = gap region (g(x) > τ)
                              │
       ░░░░░░                 │
       ░░R1░░     ●●●         │
       ░░░░░░    ●● ▲         │
                  ●●●         │
                              │
              ●●               ░░░░
            ●●●●               ░R2░    ──► UMAP-1
            ▲ ●●               ░░░░
```

`R1` and `R2` are the gap regions: surface-dense, eval-empty. The Generator targets these
first.

### 9.2 Walkthrough

| Step | What happens |
|---|---|
| 1 | Open the agent's repo. The shipped eval suite has 18 cases, 92% pass rate. The agent's tool schemas, prompts, and docstrings declare a wider conceptual surface. |
| 2 | Run `kartograf map --target python/agents/customer-service`. The decision log streams: scan_repo → ~150 surfaces, invoke_mapper, load_evals, invoke_auditor → coverage. |
| 3 | Open the map in a browser. A dense cluster of orange (surfaces) with no cyan (evals) on it is immediately visible. Hover: refund-related tool descriptions and prompt sentences. The hole is literally on screen. |
| 4 | Generator runs: real candidate count, real rejections with reasons shown, real acceptances. Click an accepted candidate: see the surfaces it was grounded in and the proposed BDD eval case. |
| 5 | Merge new evals. Re-run. Coverage updates. The map redrawn: the surface cluster from step 3 is now overlaid with eval markers. |
| 6 | The CI gate (`scripts/ci_coverage_gate.py`) reads `coverage_state` over MCP, fails the build if coverage regressed against baseline, and posts the top-3 gap regions as a PR comment. |

The 2D map carries the story. The math holds it up.

---

## 10. Why This Works

| Aspect | How Kartograf delivers |
|---|---|
| Autonomous agent | Root orchestrator with three specialized sub-agents (Mapper, Auditor, Generator) and a shared state object |
| ADK-native | Root agent, three sub-agents, shared state, all in ADK |
| MCP integration | Two roles: client (`repo_reader`, `eval_io`) and provider (`coverage_state`). Six reference MCP servers shipped. |
| Decisions, not steps | Orchestrator decides surface kinds, KDE bandwidth, gap targeting, candidate counts, validation thresholds, termination — every decision logged |
| Real signal | Turns "did I write enough tests?" from a vibes question into a metric: integrated gap mass over the agent's own declared surface area, with proposed fixes attached |

The differentiator: **the math runs on the repo alone**. No corpus to source, no dataset
to license, no production traffic to capture. The product is installable on any ADK agent
in one command and produces a useful map in under five minutes.

---

## 11. Honest Risks and Limits

- **Kartograf measures surface-vs-eval co-location, not agent correctness.** A region with
  good eval density may still contain a wrong agent. Behavioral correctness is L2 in the
  Coverage Stack.
- **Surface quality is the ceiling.** An agent with thin tool descriptions and no
  docstrings produces a small map and an easy 100% coverage. Kartograf reports surface
  count alongside coverage so this is never ambiguous, but it cannot manufacture surfaces
  the developer did not declare.
- **The 2D projection has well-known distortions.** UMAP can place semantically distant
  points near each other in 2D when the high-d manifold folds. Kartograf mitigates this by
  doing **retrieval in the original 768D space** (only KDE runs in 2D), and by reporting
  the trustworthiness score of the projection per audit.
- **Generated cases can be low-quality.** Generator validates each candidate against
  redundancy and off-spec filters. Final cases are surfaced for developer review, never
  silently committed. Propose-then-merge, not auto-commit.
- **Embedding model bias.** `text-embedding-004` is general-purpose and embeds natural
  language well, including the natural-language surfaces of an agent. It does not embed
  raw code well — which is fine because Kartograf embeds **declared surfaces** (which are
  natural language by construction), not function bodies.
- **Self-consistency, not world-coverage.** Kartograf measures whether tests exercise what
  the agent claims, not whether the agent claims enough. A spec-incomplete agent passes
  Kartograf and still ships bugs in domains it never declared. This is a feature: Kartograf
  measures one thing, measures it well, and refuses to claim the rest.
- **Sparse-surface lower bound.** A 3-tool agent with one-line descriptions produces ~10–20
  surface points. Gaussian KDE on that point cloud is unstable, gap regions become noisy,
  and the visual map has no story to tell. Kartograf detects this case at scan time
  (`|S| < 30`) and refuses to produce a coverage score, returning instead a *"surface budget
  too thin"* warning with a recommendation to expand tool descriptions or add docstrings.
  Better to refuse a number than to fabricate one.
- **k=2 worst case for generation.** When a gap region sits on the edge of surface support,
  the k-NN retrieval may return only 2 surfaces — too thin a context for a good test
  proposal. The Generator detects $k_{\text{effective}} < 3$ and marks the candidate as
  "low-confidence", surfaced separately from the high-confidence batch in the report.

---

## 12. The Coverage Stack

Kartograf is the first layer of a complete agent quality stack.

| Layer | What it answers | Status |
|---|---|---|
| **L1 — Surface coverage** | Does the eval suite exercise what the agent claims to do? | **Shipped** |
| L2 — Behavioral coverage | Does the agent reason correctly inside each surface? | Roadmap |
| L3 — Trajectory coverage | Do the agent's tool-call sequences match successful production paths? | Roadmap |
| L4 — Policy and release gates | Can coverage thresholds block a deploy in CI? | Roadmap |

The order is deliberate. Without L1, every higher layer evaluates the agent against a
biased sample of its own declared scope. L1 is the foundation that makes L2–L4 auditable.

---

## 13. Tech Stack and Architecture

### Stack overview

| Layer | Component | Purpose |
|---|---|---|
| Orchestration | Google Agent Development Kit (ADK) | Root agent, three sub-agents, shared state |
| Reasoning | Gemini 2.5 Pro | Orchestrator decisions, generation, validation |
| Embeddings | Gemini `gemini-embedding-001` (output_dim=768) | Natural-language vectorization of surfaces and evals |
| Tool protocol | Model Context Protocol (MCP) | Two roles: client (`repo_reader`, `eval_io`) and provider (`coverage_state`) |
| Projection | UMAP | 2D semantic map |
| Density estimation | scipy Gaussian KDE | Surface and eval fields, gap field |
| Gap detection | scipy connected components on $g > \tau$ | Region extraction |
| Visualization | Plotly (interactive HTML) | The 2D map; the deliverable |
| AST extraction | `ast` (stdlib) | Surface harvesting |
| CLI | typer, rich | `kartograf version`, `kartograf map` |
| Eval execution | ADK `adk eval` CLI | Re-runs target agents against the expanded suite |

### Module layout

```
cartograph/
├── core/             shared math: embeddings (Gemini), 2D reducer
├── surfaces.py       AST-based surface harvesting from a Python repo
├── evals.py          eval input extraction (.evalset.json, .test.json)
├── density.py        KDE + gap field + connected-component gap detection
├── generator.py      Gemini-grounded BDD eval case synthesis with validation
├── pipeline.py       end-to-end orchestration
├── report.py         self-contained interactive HTML (Plotly)
└── cli.py            kartograf version | map
```

---

## 14. Roadmap

The same agent can audit eval suites for any framework where surfaces and evals can be
extracted: LangChain agents (tool decorators + system prompts), CrewAI crews (role
descriptions + task definitions), OpenAI evals (system messages + ideal responses), or
any custom YAML/JSONL agent definition. Each new framework needs one `repo_reader`
adapter — no rewrite.

A natural deployment shape is the 2D map as a hosted artifact: push a commit, get a URL
to a fresh map of the agent. Diff two maps across commits to see which surfaces a
refactor added or removed and which gaps opened. The coverage score becomes a
pull-request check; the map becomes a review surface engineers actually want to look
at, not another dashboard they ignore.

The thesis the visual makes inevitable: **eval suites should be reviewed the way design
mocks are reviewed — by looking at them.** Pass rates do not lend themselves to glance
inspection; a 2D semantic map does. Kartograf is the first tool that lets a reviewer
say *"the test suite has a hole in the upper-right"* and have everyone in the room agree
on what that means.
