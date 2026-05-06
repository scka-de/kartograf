# Kartograf — User Guide

A didactic walkthrough of what Kartograf is, what every element of the map means, and
how to use the product end-to-end. Read this if you are seeing the HTML map for the
first time.

---

## Table of contents

1. [What Kartograf is, in one paragraph](#1-what-kartograf-is-in-one-paragraph)
2. [The example you are about to run](#2-the-example-you-are-about-to-run)
3. [Inputs — what Kartograf consumes](#3-inputs--what-kartograf-consumes)
4. [Running it](#4-running-it)
5. [How the map is built](#5-how-the-map-is-built)
6. [How to read the map](#6-how-to-read-the-map)
7. [What is UMAP](#7-what-is-umap)
8. [How gap regions are identified](#8-how-gap-regions-are-identified)
9. [Reading a proposed BDD eval case](#9-reading-a-proposed-bdd-eval-case)
10. [Acting on the report](#10-acting-on-the-report)
11. [Tuning knobs](#11-tuning-knobs)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What Kartograf is, in one paragraph

Kartograf is a tool that **looks at an AI agent and measures whether its tests cover
what the agent claims to do**. It draws a 2D map of the conceptual universe the agent
declares to handle (every tool description, prompt sentence, docstring) and shows where
in that universe there are **holes** — regions the agent says it can do, but no eval
case exercises.

Analogy:

> Imagine a world map where every city is a concept the agent knows how to handle
> (cancel an order, recommend a plant, schedule a service). You paint the cities blue
> if at least one test reaches them. Cities with no blue around them are unexplored
> territory: the agent claims the city exists, but no one verified it can actually
> get there.

---

## 2. The example you are about to run

The bundled ADK sample `customer-service` simulates a customer support agent for the
fictional retailer "Cymbal Home & Garden". It declares behaviors like:

- Conversing with returning customers
- Recommending plants, fertilizer, pots
- Approving discounts (with rules: up to 10% off)
- Scheduling gardening services
- Reading carts and purchase history
- Asking the customer for video to identify plants

Google's team shipped **12 eval cases** for this agent (queries like *"hi"* and
*"tell me what is in my cart?"*). Kartograf will answer: *do those 12 tests cover
everything this agent claims to handle?*

A second example is `financial-advisor` — a much richer agent with **246 declared
surfaces** and only **1 eval case**. The two examples together show the spectrum:
healthy-but-imperfect (`customer-service`, ~91% coverage) versus severely
under-tested (`financial-advisor`, ~37% coverage).

---

## 3. Inputs — what Kartograf consumes

Only the path to the agent's directory. Nothing external.

Kartograf opens every `.py` and every eval JSON file under `--target` and extracts
two sets of texts:

### Set A — Surfaces

Things the developer wrote to **declare what the agent does**:

| Kind | Where it comes from | Example from `customer-service` |
|---|---|---|
| **prompt** (purple) | Top-level string constants matching `*INSTRUCTION*` / `*PROMPT*` / `*SYSTEM*`, split into sentences | *"Approve discounts up to 10%"* |
| **tool** (orange) | First paragraph of each public function's docstring | *"Sends a link to the user's phone number to start a video session"* |
| **tool_arg** (yellow) | Each entry of the `Args:` block of a tool's docstring | *"phone_number: The phone number to send the link to"* |
| **docstring** (gray) | First paragraph of any other public class/function | *"Represents a customer's address"* |

### Set B — Eval inputs

The user-facing query of each eval case. Kartograf supports three eval-file schemas:

- ADK new schema: `{ "eval_cases": [ { "conversation": [...] } ] }`
- ADK old schema: `[ { "name": ..., "data": [ { "query": ... } ] } ]`
- Simple test schema: `[ { "query": ..., "reference": ... } ]`

For `customer-service`: 12 eval inputs. For `financial-advisor`: 1.

---

## 4. Running it

```bash
kartograf map \
  --target data/adk_samples/python/agents/customer-service \
  --output map.html

open map.html
```

Output:

```
coverage: 0.907
surfaces: 91, evals: 12, gap regions: 2
html: map.html
  R1: mass=0.069
  R2: mass=0.024
```

The HTML opens in any browser. No build step, no server. Everything is one file
(~150 KB).

If your agent has fewer than 30 declared surfaces, Kartograf refuses to score and
returns:

```
SurfaceBudgetError: surface budget too thin: 12 < 30.
Add tool descriptions, prompt content, or public docstrings before auditing.
```

This is intentional. With fewer than ~30 points the kernel density estimate is
unstable, so a coverage number would be a fabrication.

---

## 5. How the map is built

Five steps:

```
1. Each surface and eval text becomes a 768-dimensional vector  (Gemini embedding-001)
2. UMAP projects all 103 vectors into ℝ²                        (two coordinates per text)
3. Each surface becomes a colored dot
4. Each eval becomes a cyan diamond on the same plane
5. A "gap field" is computed across the plane and overlaid as a red heatmap
```

The 2D plane is what you see. The 768-dimensional space is used internally for the
nearest-neighbor retrieval that grounds each generated test.

---

## 6. How to read the map

Looking at the screen:

### Axes

`UMAP-1` and `UMAP-2` have no human unit. They are not "time" or "price". They are
two coordinates UMAP picked so the structure of the data is visible. **Only relative
distance matters** — points close together = texts that are semantically similar;
points far apart = texts that differ.

### Colored dots — the 91 surfaces

Hover any dot to see its full text and source location.

- **Purple** (`#a78bfa`) — prompt sentence
- **Orange** (`#f59e0b`) — tool description
- **Yellow** (`#fbbf24`) — tool argument description
- **Gray** (`#94a3b8`) — class or utility docstring

### Cyan diamonds — the 12 evals

Larger than the dots so they stand out. Hover for the eval text.

### Red regions — the gaps

Red translucent shapes, dotted rectangles, and labels like `R1`, `R2`. These are
the regions where surfaces are dense but evals are sparse — the gaps.

**What to look for visually**: a cluster of orange or purple dots with NO cyan diamonds
nearby. That is where the agent declares behavior the test suite never reaches.

### Sidebar

- **Coverage** as a big number (e.g. *90.7% coverage*).
- **Stats**: surface count, eval count, gap-region count, α (mass-balance), bandwidth.
- **Top gap regions**: each region listed with its 5 nearest surfaces and a generated
  BDD eval case.

---

## 7. What is UMAP

**UMAP** stands for *Uniform Manifold Approximation and Projection*. It is a
dimensionality-reduction algorithm.

Each text starts as a vector of **768 numbers** — the semantic fingerprint produced
by Gemini's embedding model. The human eye does not see 768 dimensions. UMAP
compresses those 768 numbers into 2, **trying to preserve neighborhood**: texts
that were close in 768D stay close in 2D; texts that were far stay far.

UMAP is similar to t-SNE and PCA but generally preserves both global structure
(distinct clusters look distinct) and local structure (similar points stay glued).
That is why it has become the default for visualizing embeddings.

**Distortion to be aware of**: UMAP is non-invertible — you cannot pick a coordinate
in 2D and decode it back to text. Two points that look close in 2D might in fact be
distant in 768D, similar to how Alaska and Chukotka look near each other on a flat
world map but are not. Kartograf mitigates this by doing the **gap-to-text retrieval
in the original 768D space**, not in 2D. Only the visual map and the density
estimate live in 2D.

---

## 8. How gap regions are identified

The mathematical core of the product, in three steps.

### Step A — Estimate two density fields

Imagine smearing each colored dot as a small Gaussian cloud. Sum all surface clouds:
you get **ρ_S(x, y)**, a field that says *"how much surface exists at this
coordinate"*. Do the same with the eval diamonds: **ρ_E(x, y)**.

This procedure is called **KDE** (Kernel Density Estimation).

### Step B — Compute the gap field

```
g(x, y) = ρ_S(x, y) − α · ρ_E(x, y)
```

In words: *"where there is surface AND no eval"*.

`α` corrects for the imbalance between surface count and eval count. With 91
surfaces and 12 evals, without correction the eval side would be tiny everywhere.
`α = 91 / 12 ≈ 7.6` rebalances. The sidebar shows the chosen `α`.

### Step C — Find islands above a threshold

Mark every cell where `g(x, y) > τ` (threshold = 75th percentile of `g`). Connected
red cells form an **island** = one gap region. Each gets an ID `R1`, `R2`, … and is
ranked by **integrated mass**: ∫ g dA.

### Coverage score

```
coverage = 1 − (∫ g dA) / (∫ ρ_S dA)
```

A single number from 0 to 1: the fraction of declared surface mass that has eval
mass nearby.

### Why integrated mass and not area or peak

Two regions can have very different shapes — one small but intense, one large but
mild — and have similar total *gap mass*. Integrated mass is the honest comparison;
it ranks by *"how much declared behavior is uncovered here"*, weighing size by
intensity.

---

## 9. Reading a proposed BDD eval case

For each gap region, Kartograf's generator:

1. Takes the centroid of the region in 2D
2. Retrieves the **k nearest surfaces in 768D** (default k = 5)
3. Sends the surfaces, the agent's full system prompt, and the existing eval queries
   to **Gemini 2.5 Flash**
4. Asks for **one BDD eval case** that exercises the conceptual intersection of
   those surfaces
5. Validates the result: rejects if the new query overlaps too much with an existing
   eval (redundancy filter) or if it is too short (off-spec filter)

The output looks like:

```
Scenario:   Identify unknown plant, recommend products & services
Given:      Customer lives in Las Vegas, NV; cart is empty, no prior pink-flowering
            plant purchases.
When:       I saw this amazing plant with vibrant pink flowers thriving in full sun
            at a friend's place right here in Las Vegas. I'd love to grow something
            similar, but I have no idea what it's called. Can you help me identify
            it and tell me what I'd need to take care of it?
Then:       Agent acknowledges description and location; agent requests video or
            photo for precise identification; tool call:
            product_recommendations(plant_type_description='sun-loving pink
            flowering plant', ...)
Validation: accepted
```

The **When** clause is the *actual user input you can paste into your eval suite*.
The **Then** clause is the expected agent behavior — tool calls by name, response
shape — that you assert against. The **Scenario** is the human label for the
test case.

### Validation states

- `accepted` — passed both filters; ready to merge
- `rejected_redundant` — the proposed query overlaps too much with an existing eval
- `rejected_off_spec` — the proposed query is too short or otherwise fails sanity
- `fallback_stub` — the LLM call failed (network, API key, quota); a deterministic
  interpolation is shown instead and should not be used as-is

Each proposal also carries `grounded_in`: the IDs of the surfaces it was derived
from, so you can trace any test back to the declared behaviors that motivated it.

---

## 10. Acting on the report

Typical workflow:

1. **Look at the map.** See where the gap regions are visually.
2. **Read the surrounding surfaces** in the sidebar. Understand which behavior is
   uncovered.
3. **Decide if it matters.** Sometimes the region is intentionally out of scope —
   ignoring it is fine.
4. **Copy the BDD case** into your eval format (ADK `.evalset.json`, pytest, YAML,
   whatever you use). The `When` becomes the query, the `Then` becomes the
   reference / expected tool calls.
5. **Re-run** `kartograf map`. The new eval should fall inside the previously red
   region. The region either disappears or shrinks.

### How this differs from line coverage

Tools like `coverage.py` and `gcov` answer: *"line 42 of `tools.py` was never
executed"*. Kartograf answers: *"the intersection between 'apply discount' and
'privacy request' was never tested conceptually"*. Two complementary questions —
both legitimate, only one of them existed before.

---

## 11. Tuning knobs

Defaults are sensible. Override per-audit if needed:

| Flag | Default | Effect |
|---|---|---|
| `--alpha <float>` | `|S|/|E|` (mass-balanced) | Use `1.0` for raw-density gap (treats every point equally) |
| `--k <int>` | `5` | Number of surfaces retrieved per gap region for BDD generation |
| `--output <path>` | `kartograf_map.html` | Where to write the HTML report |

Surface budget floor is hardcoded at 30 in `cartograph/pipeline.py`. Below that
Kartograf refuses to score; this is intentional and not currently user-tunable.

---

## 12. Troubleshooting

### `SurfaceBudgetError`

Your agent declares fewer than 30 surfaces. Two paths:

- **Enrich the agent**: write longer tool descriptions, add docstrings to public
  functions, expand the system prompt. Re-run.
- **Audit a different agent**: this one's declared contract is too thin for the
  math to be meaningful.

### Coverage looks artificially high

You probably do not have `GOOGLE_API_KEY` set, or the Gemini API is not enabled in
your GCP project. Kartograf falls back to a deterministic local embedder that does
not separate semantically — so all texts cluster together, and gaps disappear.

Check:

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); \
  print('key set:', bool(os.getenv('GOOGLE_API_KEY')))"
```

Enable the API:

```bash
gcloud services enable generativelanguage.googleapis.com
```

Then clear the cache and re-run:

```bash
rm -rf data/embeddings/kartograf_audit_*.npy
kartograf map --target ... --output ...
```

### BDD proposals show `fallback_stub`

The Gemini *generation* call (separate from embeddings) failed. Common causes:
quota, model name change (Gemini retires models periodically), expired key.
Check the model in `cartograph/generator.py:GENERATION_MODEL`. Available models
can be listed via:

```python
from google import genai, os
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
for m in client.models.list():
    if 'generateContent' in (m.supported_actions or []):
        print(m.name)
```

### Map looks crowded

Try a smaller agent, or open the HTML and use Plotly's lasso/zoom to focus on a
region. Automated zoom-to-region is on the roadmap.

---

## Where to next

- The proposal: [docs/proposal.md](docs/proposal.md) — full product thesis, math,
  demo plan.
- Hero artifacts:
  - [docs/figures/kartograf_map_customer_service.html](docs/figures/kartograf_map_customer_service.html)
  - [docs/figures/kartograf_map_financial_advisor.html](docs/figures/kartograf_map_financial_advisor.html)
- Fleet of validated runs across other ADK samples:
  `docs/figures/fleet/*.html`.
