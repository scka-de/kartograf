# Cartograph

> **Submission for the Google for Startups AI Agents Challenge — Track 1: Build**
> *Cartograph maps the world your agent actually faces.*

---

## 1. One-Sentence Pitch

Cartograph is an autonomous ADK agent that audits the eval suite of any other agent — mapping the real semantic world the agent must handle, identifying the regions where existing evals don't represent that world, and proposing new test cases grounded in real corpus exemplars.

> **Cartograph doesn't tell you whether your agent is correct everywhere. It tells you whether your tests are looking in the right places.**

To prove it works, Cartograph applies itself to five of Google's own ADK sample agents and reports their semantic coverage as a public benchmark.

---

## 2. The Problem

Agents pass evals and break in production because no one measures whether the evals represent the world the agent actually faces.

A typical team writes 15–30 hand-crafted test cases, hits 90%+ pass rate, and ships. Two weeks later, production users hit cases the team never imagined: ambiguous requests, multi-intent messages, edge cases the eval suite never tested, error states the agent has no exemplar for.

The eval suite scored the agent against the team's imagination, not against the observed world.

| Existing tools | What they measure |
|---|---|
| LangSmith, Braintrust, Langfuse | Pass rate of an agent against a given suite |
| Patronus, Galileo, Arize Phoenix | Runtime guardrails and observability |
| Synthetic test generation papers | Tests generated in a vacuum, not anchored to a real corpus |

**None of them ask: is this eval suite a faithful map of the production input distribution?**

That is the gap Cartograph fills.

---

## 3. The Insight

Software engineers solved an analogous problem 30 years ago for code: `coverage.py`, `gcov`, `istanbul`. They measure whether tests touch the lines, branches, and paths a program contains.

Agent evaluation has no equivalent. There is no tool that says: *"your eval suite represents 41% of the semantic regions your agent has been observed to face."*

Cartograph is that tool — but for the semantic surface of an agent's task space, not for source code lines.

---

## 4. Track Fit

**Track 1: Build (Net-New Agents).**

Cartograph is a net-new autonomous agent built on ADK. It uses Gemini for embeddings and reasoning, exposes its capabilities through MCP, and securely connects to external eval repositories, corpus stores, and target agents to gather context and act autonomously.

It is not a static pipeline. The Cartograph orchestrator decides clustering strategy based on corpus characteristics, prioritizes coverage gaps by risk, chooses which uncovered regions to generate tests for, validates generated cases against the corpus before proposing them, and writes new eval files into the target repo via MCP. Each of these is a *decision*, made at runtime, based on the current coverage state — not a hardcoded step.

This is "acts, not just responds" in the literal sense.

---

## 5. How Coverage Is Measured

Semantic coverage is defined precisely so it can be audited and reproduced.

**Coverage score:**

$$\text{coverage} = \frac{\sum_i \text{density}_i \cdot \text{covered}_i}{\sum_i \text{density}_i}$$

where:
- $\text{density}_i$ = fraction of the real-world corpus that falls in region $i$
- $\text{covered}_i = 1$ if region $i$ contains at least one eval case projection, else $0$
- regions are clusters produced by HDBSCAN over Gemini embeddings of the corpus

A region holding 8% of corpus mass with zero eval cases drags the score down by 8 points. A region holding 0.1% of mass barely moves the needle. Coverage is weighted by what the agent is *observed* to face, not by the count of distinct topics.

In v1, $\text{covered}_i$ is binary at the region level for interpretability — a region either has at least one eval projection or it does not. The report also surfaces eval *density* per region, so a region with one eval and thousands of corpus examples is visibly under-tested even when marked covered. A weighted variant of $\text{covered}_i$ is on the v2 roadmap.

**Risk score for uncovered regions:**

$$\text{risk}_i = \text{density}_i \times \text{heterogeneity}_i \times \text{distance\_to\_nearest\_eval}_i$$

where heterogeneity is the average pairwise embedding distance among corpus examples in the region, and distance is from region centroid to the closest existing eval case in embedding space.

The orchestrator uses risk score, not raw region size, to rank gaps. A small but heterogeneous region far from any existing test is more dangerous than a large homogeneous region that is "almost covered."

**What this metric does and does not claim:** the score measures eval-suite *representativity* with respect to an observed corpus. It does not certify production readiness, agent correctness, or any guarantee about unobserved inputs. The corpus is itself a sample, and the score is meaningful only relative to that sample.

---

## 6. How Cartograph Works

Cartograph is a multi-agent ADK system with a root orchestrator and three specialized sub-agents.

The **root orchestrator** is the Cartograph agent itself. It is LLM-driven (Gemini 2.5 Pro) and decides what to do next based on the current coverage state. The flow is not hardcoded — for an agent already at high coverage, the orchestrator may skip Generator entirely. For an agent with concentrated gaps in a few high-risk regions, it may run Generator multiple times targeting different regions before terminating.

The **three sub-agents** each have a single, well-scoped responsibility:

- **Mapper.** Ingests the real-world corpus. Embeds with Gemini `text-embedding-004`. Clusters into semantic regions using HDBSCAN; UMAP is used for 2D visualization, and for the clustering itself we use the original embedding space or a moderate UMAP reduction (the target dimensionality is a tuned parameter logged per audit). Produces a labeled map of the world the target agent must handle.
- **Auditor.** Ingests the existing eval suite. Projects each eval case into the same semantic space. Computes per-region coverage using the metric above, identifies uncovered regions, flags redundant tests, ranks gaps by risk score.
- **Generator.** Takes the top-K uncovered regions selected by the orchestrator. Retrieves representative real examples from the corpus, generates new eval cases grounded in those examples, validates each generated case against two filters (redundancy with existing evals, distance from real corpus exemplars), and proposes accepted cases for developer review.

The three sub-agents communicate through a shared coverage state object that the orchestrator reads to make decisions and that external systems can query through MCP.

### Sample orchestrator log

The orchestrator's reasoning is exposed as a structured log. A representative excerpt from a single audit run:

```
step 1  ingest        → corpus: 27,113 Bitext interactions
                        eval suite: 18 cases (adk-samples customer-service)

step 2  invoke Mapper → 11 regions detected
                        density [0.18, 0.14, 0.11, 0.09, 0.08, ...]

step 3  invoke Auditor → coverage = 0.41
                         5 uncovered regions ranked by risk:
                           refund_with_cancellation       (risk 0.087)
                           multi_intent_complaint         (risk 0.061)
                           escalation_after_unresolved    (risk 0.044)
                           billing_dispute_with_evidence  (risk 0.029)
                           privacy_request_inside_support (risk 0.022)

step 4  decide        → coverage 0.41 < threshold 0.75
                        invoke Generator on top 2 regions
                        target: 6 candidates per region

step 5  invoke Generator → 14 candidates produced
                           2 rejected: redundant with existing eval cases
                           1 rejected: distance to nearest exemplar above 0.42
                           11 accepted, grounded in 27 specific Bitext exemplars

step 6  invoke Auditor → coverage = 0.57

step 7  decide        → coverage 0.57 < threshold 0.75
                        remaining uncovered regions are low-risk
                        terminate, surface report for developer review
```

Thresholds (coverage target, distance cutoff, candidates per region) are exposed as policy parameters with documented defaults. The orchestrator selects which regions to target, how many candidates to generate per region, when to terminate, and which candidates to reject — every decision is logged for reproducibility. This log is the artifact that makes Cartograph an agent rather than a pipeline.

*See `architecture.svg` for the full system diagram and `audit_lifecycle.svg` for the run sequence.*

---

## 7. How MCP Fits

MCP plays two distinct roles in Cartograph, and the distinction matters.

**Cartograph as MCP client** — consumes MCP servers exposed by the customer's environment:

- `corpus_reader` — exposes a real-world corpus (tickets, conversations, code issues, traces) as a queryable source
- `eval_io` — reads existing eval suites and writes new ones to wherever they live

**Cartograph as MCP provider** — exposes its own MCP server for the rest of the customer's stack:

- `coverage_state` — exposes the current semantic coverage map and gap report as a first-class signal that other systems can query

The provider role is what makes Cartograph plug-and-play in a larger agent quality stack. CI pipelines, monitoring dashboards, the future L2/L3/L4 layers (see The Coverage Stack) — all of them read coverage state via MCP.

### Who implements the MCP servers?

| Server | Implemented by | Why |
|---|---|---|
| `coverage_state` | **Cartograph** | This is the canonical product output. There is one correct way to expose coverage state, and Cartograph defines it. |
| `corpus_reader` | Cartograph ships **reference servers** (Bitext, GitHub issues, StackExchange, ADK eval format, generic JSONL). Customers extend or replace for proprietary corpora. | Corpus sources are too varied to standardize. Reference implementations remove friction for 80% of cases. |
| `eval_io` | Cartograph ships **reference servers** (Git repo, ADK `.evalset.json`, generic YAML/JSONL). Customers extend for proprietary eval databases. | Same reason. |

This is the same model Anthropic uses for MCP itself: provide canonical reference servers for common cases, expose extension points for custom integrations. Cartograph's v1 ships with five reference servers plus the canonical `coverage_state` provider.

---

## 8. What Cartograph Outputs

Every audit produces three artifacts.

**1. Coverage report.** A single number (semantic coverage score) plus a per-region breakdown showing density, coverage status, and risk score. This is the headline answer to "is this eval suite representative?"

**2. Top uncovered regions.** A ranked list of semantic regions the eval suite does not represent, with corpus exemplars from each. Developers see exactly what kind of input is missing — not just "category X is uncovered" but "your tests have no cases that look like these specific 5 real interactions."

**3. Proposed eval cases.** New test cases generated by Cartograph, grounded in real corpus exemplars, formatted in the target framework's eval format (ADK `.evalset.json`, generic YAML, JSONL). Cases are surfaced for developer review — not committed automatically. Each case includes provenance: which corpus exemplars it was grounded in, which region it covers, why the Generator considers it non-redundant.

The three together answer the questions a team needs answered before shipping: how representative are my tests, where are the holes, and what would close them.

---

## 9. Demo Plan

The demo applies Cartograph to **five agents from `google/adk-samples`**, using domain-matched public corpora.

**The framing matters and we say it explicitly on stage:** ADK samples are intentionally educational reference implementations, not production-certified deployments. Their eval suites are illustrative by design. Cartograph quantifies the distance between an educational eval suite and a production-representative one — a distance that exists for *every* eval suite ever shipped, including those of paying customers. The demo is not a critique of the samples; it is a measurement of a gap that always exists.

### Agents audited

| ADK sample agent | Public corpus | Domain |
|---|---|---|
| `customer-service` | Bitext customer support (~27k labeled interactions) | Support / e-commerce |
| `software-bug-assistant` | StackOverflow Python tag + curated GitHub issue dumps | Code / debug |
| `financial-advisor` | Personal Finance StackExchange | Finance Q&A |
| `travel-concierge` | Travel StackExchange | Travel Q&A |
| `data-science` | Kaggle Discussions + StackOverflow data-science tag | Data / analytics |

Each agent's existing `.evalset.json` files (the ADK Golden Datasets shipped with the samples) become the eval suite under audit. The matched public corpus becomes the observed real-world input distribution. Cartograph runs the same audit pipeline on all five.

### Demo arc (4 minutes)

| Time | Beat |
|---|---|
| 0:00–0:30 | **Framing.** Five ADK sample agents, all passing their own evals at 90%+. Pose the question: "what does the real input distribution look like, and how much of it do these tests represent?" State the educational-vs-production framing explicitly. |
| 0:30–1:00 | **Fleet audit.** Cartograph runs on all five. Headline table: pass rate next to semantic coverage for each agent. Pass rates uniformly high, coverage scores uniformly lower. The gap is the story. |
| 1:00–2:00 | **Deep dive on `customer-service`.** Show 2D projection of Bitext intents — the observed world. Project the 18 eval cases onto it; entire regions are red. Show the orchestrator's decision log live: detected 5 uncovered regions, ranked by risk, decided to invoke Generator on the top 2. |
| 2:00–2:40 | **Generator validation visible.** 14 candidates produced; 2 rejected as redundant with existing evals; 1 rejected as too far from corpus exemplars; 11 accepted, each linked to specific Bitext exemplars. The rejection step is shown explicitly — this is corpus-grounded synthesis with validation, not blind generation. |
| 2:40–3:30 | **The aha moment, framed honestly.** Re-run `adk eval` against the expanded suite. Side-by-side reveal: |
| | • **Before:** 92% pass rate, 18 cases, **41% semantic coverage** |
| | • **After:** 68% pass rate, 29 cases, **57% semantic coverage** |
| | The framing line on screen: *"the agent didn't get worse — the test got more honest. The 92% was measuring a smaller world."* |
| 3:30–4:00 | **Close.** Show the orchestrator decision log across all five agents — different decisions for different agents (skipped Generator on one, ran twice on another). Final line: *"Cartograph doesn't tell you whether your agent is correct everywhere. It tells you whether your tests are looking in the right places."* |

The fleet benchmark and corpus embeddings are precomputed (deterministic from the same pipeline), allowing the deep-dive on `customer-service` to execute live within the four-minute window — Generator producing cases, Auditor re-projecting onto the semantic map, and a re-run of `adk eval` against the expanded suite. Coverage scores, generated cases, and pass-rate deltas are all produced by the same pipeline that runs in production.

---

## 10. Why This Wins on Track 1's Criteria

| Track 1 requirement | How Cartograph delivers |
|---|---|
| Net-new autonomous agent | Multi-agent ADK system with LLM-driven root orchestrator and three specialized sub-agents, built from scratch |
| Built with ADK | Yes — root agent, three sub-agents, shared state, all in ADK |
| Uses MCP to connect to external tools | Two MCP roles: client (corpus_reader, eval_io) and provider (coverage_state). Five reference MCP servers shipped with v1. |
| Gathers context and executes tasks autonomously | Root agent decides clustering strategy, gap prioritization, generation targets, validation thresholds, and termination — without human intervention. The decision log is exposed as a first-class artifact. |
| Drives real business results | Turns eval-suite representativity from an unmeasured assumption into a quantified, auditable signal that helps teams identify production-input gaps before they become incidents |

---

## 11. Honest Risks and Limits

A v1 in two weeks earns its credibility by stating what it does *not* yet do.

- **Cartograph measures eval-suite representativity, not agent correctness.** Whether the agent reasons correctly inside a covered region is out of scope — that belongs to L2 in the Coverage Stack.
- **Cartograph measures coverage of an *observed* input distribution, not of all possible production inputs.** The corpus is itself a sample. The coverage score is meaningful only relative to that sample, and customers should refresh the corpus periodically to keep the measure current.
- **Cartograph does not measure security, latency, cost, tool-failure modes, or long-trajectory behavior.** Those are orthogonal quality dimensions that require their own tooling. Cartograph is the missing coverage layer, not a complete quality stack.
- **Cluster quality depends on embedding quality.** Cartograph uses Gemini embeddings as the default. Where a domain produces low-confidence clusters, the orchestrator flags them rather than pretending they are clean.
- **Generated test cases can be low-quality.** Generator validates each case against two filters (redundancy, distance from corpus). Final cases are surfaced to the developer for review, not silently committed. The expected workflow is propose-then-merge, not auto-commit.
- **Risk scoring is heuristic in v1.** Combines region density, heterogeneity, and distance-to-nearest-eval. A v2 incorporates domain-specific risk labels (regulatory, customer-tier, revenue-impact).
- **The five-agent demo is representative, not exhaustive.** The benchmark covers five domains. Generalizing to all 30+ ADK sample agents is mechanical, not conceptual.

---

## 12. The Coverage Stack

Cartograph is the first layer of a complete agent quality stack. The other layers exist as a roadmap, not as v1 features — but naming them now clarifies why semantic coverage is the foundation, not a niche.

| Layer | What it answers | Status |
|---|---|---|
| **L1 — Semantic coverage** | Does the eval suite represent the world the agent will face? | **v1 — this submission** |
| L2 — Behavioral coverage | Does the agent reason correctly inside each represented region? | Roadmap |
| L3 — Trajectory coverage | Do the agent's tool-use sequences match successful production trajectories? | Roadmap |
| L4 — Policy and release gates | Can coverage thresholds block a deploy in CI? | Roadmap |

The order is deliberate. Without L1, every higher layer evaluates the agent against a biased sample of reality. You cannot trust a behavioral judge on a region the eval suite never tested, and you cannot gate a release on a coverage signal that does not exist. Semantic coverage is the foundation that makes the rest auditable.

This is also the natural Track 3 path: each higher layer becomes a Marketplace-ready module on top of the same coverage primitive, all of them reading from the same `coverage_state` MCP provider.

*See `coverage_stack.svg` for the layered roadmap.*

---

## 13. Tech Stack and Architecture

### Stack overview

| Layer | Component | Purpose |
|---|---|---|
| Orchestration | Google Agent Development Kit (ADK) | Root agent, three sub-agents, shared state |
| Reasoning | Gemini 2.5 Pro | Orchestrator decisions, clustering strategy, gap prioritization, test generation, validation |
| Embeddings | Gemini `text-embedding-004` | Semantic vectorization of corpus and eval cases |
| Tool protocol | Model Context Protocol (MCP) | Two roles: client (corpus_reader, eval_io) and provider (coverage_state) |
| Vector index | Vertex AI Vector Search | Persistent semantic index over the corpus |
| Clustering | UMAP + HDBSCAN | 2D projection and density-based region detection |
| Eval execution | ADK `adk eval` CLI | Re-runs target agents against the expanded eval suite |
| Deployment | Cloud Run | Containerized agent deployment, scales to zero between audits |
| Observability | Cloud Logging + OpenTelemetry | Audit trail of orchestrator decisions for reproducibility |

### Reference MCP servers shipped in v1

| Server | Role | Purpose |
|---|---|---|
| `corpus_reader_bitext` | Client adapter | Reads Bitext customer support corpus |
| `corpus_reader_stackexchange` | Client adapter | Reads StackExchange dumps for any tag |
| `corpus_reader_github` | Client adapter | Reads GitHub issues from any public repo |
| `eval_io_adk` | Client adapter | Reads and writes ADK `.evalset.json` Golden Datasets |
| `eval_io_git` | Client adapter | Generic Git repo eval suite I/O |
| `coverage_state` | Provider | Canonical Cartograph output — exposes the coverage map |

### System architecture

The Cartograph agent has a root orchestrator that decides what to do next based on the current coverage state, three specialized sub-agents that perform concrete steps, and an MCP boundary that mediates all communication with the external world.

*See `architecture.svg` for the full system diagram.*

### Audit lifecycle

A complete audit proceeds through five stages with two orchestrator decision points: whether to invoke Generator at all (skip if coverage already exceeds threshold), and whether to run another Generator pass on a different region after writing new tests.

*See `audit_lifecycle.svg` for the run sequence.*

---

## 14. Beyond the Hackathon

Cartograph is built framework-agnostic by design — the demo uses ADK and `adk-samples` because that is the strongest narrative for this challenge, but the same agent can audit eval suites for LangChain, CrewAI, OpenAI evals, LangSmith datasets, or any custom YAML/JSONL store. The MCP boundary is what makes this true: any new framework only requires a new `eval_io` adapter, not a rewrite.

The post-hackathon path is Track 3: refactor for Google Cloud Marketplace as a managed service that any team building agents on Google Cloud can install to measure and improve eval-suite representativity continuously, with `coverage_state` piped into CI as a release gate.

The long-term vision: every agent CI pipeline shows two numbers — pass rate and coverage. Today only the first one exists. Cartograph supplies the second.

---

## 15. Submission Details

- **Track:** Track 1 — Build
- **Submitter:** Solo
- **Organization:** Station932
- **Country:** Germany
- **Funding stage:** [to be filled per form]

---
