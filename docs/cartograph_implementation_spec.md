# Cartograph — Implementation Spec

> **Audience.** This document is the complete specification for building Cartograph v1. It is written so an AI coding assistant (Claude Code, Cursor, etc.) can implement the project end-to-end without asking clarifying questions. Every ambiguity is resolved by an explicit decision in this document.
>
> **Status.** v1 — hackathon scope. Two weeks, solo developer, target submission June 5, 2026.

---

## 0. How to Use This Spec

Read in this order: §1 (vision) → §2 (deliverables) → §3 (build order) → then jump to whichever module you're working on. §13 is the master Definition of Done; §13.1 is the single-command final acceptance.

When this spec and the proposal (`cartograph_proposal.md`) disagree on a technical detail, **this spec wins**. The proposal is for judges; this spec is for builders.

When in doubt about scope, default to "v1, hackathon, demo-driven." We are not building production infrastructure. We are building a demonstrably autonomous agent that proves the concept.

---

## 1. Product Vision

Cartograph is an autonomous agent that audits eval suites of other agents. It maps the real semantic world an agent must handle (from a real-world corpus), measures how much of that world the existing eval suite represents, and proposes new eval cases grounded in real corpus exemplars. The audit is run by a multi-agent ADK system: a root orchestrator decides the flow, three specialized sub-agents (Mapper, Auditor, Generator) execute concrete steps, and Cartograph exposes its findings as an MCP server so other systems can consume coverage as a first-class signal.

The single sentence that explains what users get:

> Cartograph doesn't tell you whether your agent is correct everywhere. It tells you whether your tests are looking in the right places.

---

## 2. What We're Building (Concrete Deliverables)

The end state of v1 is one git repository containing:

1. **A Python package `cartograph`** that installs via `pip install -e .` and provides:
   - A CLI: `cartograph audit`, `cartograph report`, `cartograph demo prepare`, `cartograph demo run`
   - Three ADK sub-agents (Mapper, Auditor, Generator) and one root agent
   - Six MCP servers (5 reference adapters + 1 canonical provider)
   - Core libraries for embeddings, clustering, coverage metric, risk score, ADK eval invocation, and storage

2. **Demo infrastructure** that produces:
   - Precomputed fleet benchmark across 5 ADK sample agents
   - A live deep-dive script for `customer-service` that runs in under 90s
   - Static 2D semantic-map images for the visualization

3. **A Definition-of-Done checklist** that is fully ticked (§13), plus a `scripts/run_demo.sh` final acceptance test (§13.1).

4. **A README** that lets a fresh checkout run the smoke demo in under 10 minutes assuming `GOOGLE_API_KEY` is set and warm caches exist (precomputed JSON artifacts checked into the repo, plus first-run download of Bitext from HuggingFace, ~3 minutes). A full cold prepare from zero may take up to 30 minutes; that path is documented separately in the README.

That's it. No web UI, no Cloud Run deployment, no CI integration in v1.

---

## 3. Build Order

Strictly sequential. Each step depends on the previous.

| Order | Component | Why first |
|---|---|---|
| 1 | Repo skeleton + tooling (pyproject, pre-commit, pytest) | Everything else needs it |
| 2 | Core data models (`cartograph/core/models.py`) | All other modules import these |
| 3 | Storage (`cartograph/core/storage.py`) | SQLite schema, `.npy` and `.joblib` paths |
| 4 | Embeddings wrapper (`cartograph/core/embeddings.py`) | Gemini API, batched, cached |
| 5 | Clustering (`cartograph/core/clustering.py`) | UMAP fit/transform + HDBSCAN |
| 6 | Coverage metric + risk score (`cartograph/core/coverage.py`, `risk.py`) | Pure functions over models, all in 30d |
| 7 | ADK eval runner (`cartograph/core/adk_eval.py`) | Subprocess wrapper, parses output |
| 8 | `corpus_reader_bitext` MCP server | Smallest end-to-end input |
| 9 | `eval_io_adk` MCP server | Reads/writes `.evalset.json` |
| 10 | Mapper sub-agent | Wraps embeddings + clustering, persists reducer |
| 11 | Auditor sub-agent | Wraps coverage + risk, uses persisted reducer |
| 12 | Generator sub-agent | Most complex; needs Mapper+Auditor done |
| 13 | `coverage_state` MCP server | Reads from storage; needs storage populated |
| 14 | Root orchestrator | Wires sub-agents together |
| 15 | CLI | Glue layer |
| 16 | Other corpus readers (`stackexchange`, `github`) | Needed only for fleet benchmark |
| 17 | `eval_io_git` MCP server | Generic fallback for non-ADK evals |
| 18 | Demo precomputation pipeline | Runs steps 1-14 against 5 agents |
| 19 | Live deep-dive script | Demo recording-ready |
| 20 | Visualization (2D projection images) | Last polish |
| 21 | `scripts/run_demo.sh` + README + final integration test | Ship |

If schedule slips, cut from steps 16–17 first (other corpora) and run the fleet with only Bitext + a fake corpus for the other four agents. The deep-dive on customer-service still works because that path is fully built.

---

## 4. Tech Stack (Exact Choices)

```
Python:           3.11+
ADK:              google-adk>=1.0
Gemini SDK:       google-genai>=0.5
MCP SDK:          mcp>=1.2
Embeddings model: text-embedding-004 (768 dim)
Reasoning model:  gemini-2.5-pro
Vector similarity: numpy (NOT FAISS — see decision below)
Clustering:       hdbscan>=0.8, umap-learn==0.5.5 (PINNED for joblib compat)
Reducer persistence: joblib>=1.3
Storage:          sqlite3 (stdlib) + numpy arrays on disk + joblib for reducers
CLI:              typer>=0.12
Terminal output:  rich>=13
Data models:      pydantic>=2.5
HF datasets:      datasets>=2.18 (for Bitext)
GitHub API:       PyGithub>=2.1
StackExchange:    stackapi>=0.2
Visualization:    matplotlib>=3.8, seaborn>=0.13
Testing:          pytest>=7.4, pytest-asyncio, pytest-mock
Linting:          ruff>=0.4
Type checking:    mypy>=1.8 (strict mode for cartograph/core, loose elsewhere)
Env:              python-dotenv>=1.0
```

**Decisions explicitly made:**
- We use `text-embedding-004` (768d), not the 1536d variant.
- **Vector similarity uses NumPy directly, not FAISS.** Corpus sizes in v1 (~10k–30k items) make NumPy cosine similarity trivially fast. FAISS adds an integration that breaks more than it helps at this scale. Deferred to post-v1 if/when corpus sizes exceed ~1M.
- **Vector space convention (critical):** Gemini produces 768d embeddings. UMAP reduces to 30d. *All region-level computations* (centroids, distances, risk score, eval projection) operate in the 30d space. Visualization uses a separate 2d UMAP fit. There is no mixing of 768d and 30d in any computation. See §6 for the full convention and §7.2 for the API.
- `umap-learn` is pinned to `0.5.5` because joblib serialization compatibility across UMAP versions is fragile, and the audit lifecycle persists fitted reducers.
- Storage is SQLite for metadata + on-disk `.npy` files for embedding matrices + `.joblib` for fitted UMAP reducers.
- All MCP servers use stdio transport. No SSE in v1.
- All async where ADK requires it; otherwise sync. Don't async-everything.

---

## 5. Repository Layout

```
cartograph/
├── pyproject.toml
├── README.md
├── .env.example                 # GOOGLE_API_KEY, GITHUB_TOKEN, etc.
├── .gitignore
├── .pre-commit-config.yaml
│
├── cartograph/                  # The Python package
│   ├── __init__.py
│   ├── cli.py                   # Typer app — entry point
│   │
│   ├── agents/                  # ADK agents
│   │   ├── __init__.py
│   │   ├── root.py              # CartographAgent (root orchestrator)
│   │   ├── mapper.py            # MapperAgent
│   │   ├── auditor.py           # AuditorAgent
│   │   └── generator.py         # GeneratorAgent
│   │
│   ├── mcp_servers/             # MCP servers (each is runnable)
│   │   ├── __init__.py
│   │   ├── _common.py           # Shared MCP server scaffolding
│   │   ├── corpus_reader_bitext.py
│   │   ├── corpus_reader_stackexchange.py
│   │   ├── corpus_reader_github.py
│   │   ├── eval_io_adk.py
│   │   ├── eval_io_git.py
│   │   └── coverage_state.py
│   │
│   ├── core/                    # Pure-Python core, no ADK dependency
│   │   ├── __init__.py
│   │   ├── models.py            # Pydantic models (with vector-space conventions)
│   │   ├── embeddings.py        # Gemini embedding wrapper + cache
│   │   ├── clustering.py        # UMAP fit/transform + HDBSCAN
│   │   ├── coverage.py          # Coverage metric, noise handling
│   │   ├── risk.py              # Risk score (operates in 30d)
│   │   ├── validation.py        # Generator validation filters
│   │   ├── adk_eval.py          # Subprocess wrapper for `adk eval`
│   │   └── storage.py           # SQLite + .npy + .joblib persistence
│   │
│   └── demo/
│       ├── __init__.py
│       ├── prepare.py           # Precompute fleet benchmark
│       ├── deep_dive.py         # Live customer-service audit
│       ├── visualize.py         # 2D semantic-map images
│       └── mocks/               # Bundled fallback corpus fixtures (committed)
│           ├── corpus_reader_stackexchange.json
│           ├── corpus_reader_github.json
│           └── corpus_reader_bitext.json   # Tiny version for tests
│
├── data/                        # Created at runtime, gitignored EXCEPT data/precomputed/
│   ├── corpora/                 # Cached raw corpora (gitignored)
│   ├── embeddings/              # Cached embedding matrices (.npy, gitignored)
│   ├── adk_samples/             # Cloned google/adk-samples subset (gitignored)
│   ├── audits/                  # SQLite + per-audit artifacts (gitignored, see §7.5)
│   └── precomputed/             # Demo-ready JSON outputs (COMMITTED — small, no embeddings)
│
├── tests/
│   ├── __init__.py
│   ├── test_core_coverage.py
│   ├── test_core_clustering.py
│   ├── test_core_risk.py
│   ├── test_core_adk_eval.py
│   ├── test_core_validation.py
│   ├── test_mcp_eval_io_adk.py
│   ├── test_mcp_corpus_reader_bitext.py
│   ├── test_agent_mapper.py
│   ├── test_agent_auditor.py
│   ├── test_agent_generator.py    # Uses mocked Gemini with deliberate reject candidates
│   ├── test_orchestrator_decisions.py
│   ├── test_e2e_smoke.py          # Tagged @pytest.mark.e2e, opt-in
│   └── conftest.py                # Fixtures: tiny corpus, tiny eval suite, mock_gemini
│
└── scripts/
    ├── fetch_bitext.py
    ├── fetch_stackexchange.py
    ├── fetch_github_issues.py
    ├── clone_adk_samples.py
    └── run_demo.sh              # Final acceptance test (§13.1)
```

---

## 6. Data Models (Pydantic v2)

All in `cartograph/core/models.py`. Use these schemas verbatim.

### Vector space convention

Two embedding spaces exist throughout the codebase. After processing by Mapper / Auditor / Generator, embedded objects may carry both representations:

- **`embedding`**: the original 768d Gemini embedding (used for redundancy checks between eval cases, where we want maximum semantic fidelity).
- **`embedding_30d`**: the UMAP-reduced 30d embedding (used for ALL region-level computations: clustering, centroid distances, risk scores, eval-to-region assignment).

A freshly created `CorpusItem` or `EvalCase` has both fields as `None`; they are populated as the audit progresses. Any code path that reads `embedding_30d` MUST handle the `None` case (typically by raising — the order of operations should make `None` unreachable in production paths).

The fitted UMAP reducer is persisted per audit (`reducer_30d.joblib`) so that any embedding produced after the corpus is mapped (eval cases, generated cases) can be transformed into the same 30d space. Never compute distances between mismatched spaces.

### Models

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime

class CorpusItem(BaseModel):
    id: str                       # Stable hash of source + content
    source: str                   # e.g. "bitext", "github:org/repo"
    text: str                     # The raw text input the agent would face
    metadata: dict = Field(default_factory=dict)
    embedding: Optional[list[float]] = None        # 768d (Gemini)
    embedding_30d: Optional[list[float]] = None    # 30d (UMAP-reduced)
    region_id: Optional[str] = None                # Set after clustering; None means noise

class EvalCase(BaseModel):
    id: str
    source_path: str
    content: str                  # Concatenated input text used for embedding
    raw: dict = Field(default_factory=dict)        # Full original eval object, round-tripped
    embedding: Optional[list[float]] = None        # 768d
    embedding_30d: Optional[list[float]] = None    # 30d (transformed via persisted reducer)
    region_id: Optional[str] = None                # Assigned by Auditor
    assignment_distance: Optional[float] = None    # Cosine distance in 30d to assigned region centroid

class Region(BaseModel):
    id: str                       # e.g. "region_03"
    label: str                    # Human-readable, inferred from exemplars
    centroid: list[float]         # ALWAYS 30d (L2-normalized)
    density: float                # member_count / total_clustered_count, in [0, 1]
                                  # (excludes noise items — denominator is non-noise corpus only)
    heterogeneity: float          # Mean pairwise cosine distance among members in 30d.
                                  # Embeddings are L2-normalized before distance, then result
                                  # is clamped to [0, 1] for risk-score numerical stability.
    member_count: int             # Number of corpus items in this region (excludes noise).
                                  # Persisted alongside member_corpus_ids; both are kept in sync.
    member_corpus_ids: list[str] = Field(default_factory=list)
    exemplar_corpus_ids: list[str] = Field(default_factory=list)  # Top 5 representative
    eval_case_ids: list[str] = Field(default_factory=list)
    eval_density: float = 0.0     # eval_count / member_count for this region

class Gap(BaseModel):
    region_id: str
    risk_score: float
    components: dict = Field(default_factory=dict)  # {"density": x, "heterogeneity": y, "distance": z}
    rank: int

class GeneratedCase(BaseModel):
    id: str
    region_id: str
    content: str
    raw: dict = Field(default_factory=dict)         # Full eval object in target format
    grounded_in_corpus_ids: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None         # 768d (for redundancy check)
    embedding_30d: Optional[list[float]] = None     # 30d (for off-corpus check)
    validation_status: Literal["accepted", "rejected_redundant", "rejected_off_corpus"]
    validation_details: dict = Field(default_factory=dict)
    accepted: bool

class RedundantEval(BaseModel):
    case_id: str
    duplicate_of_case_id: str
    similarity: float             # 768d cosine similarity, > 0.85 threshold

class Decision(BaseModel):
    step: int
    timestamp: datetime
    action: str                   # "ingest", "invoke_mapper", "invoke_auditor",
                                  # "invoke_generator", "decide_continue", "decide_terminate"
    inputs: dict = Field(default_factory=dict)
    outputs: dict = Field(default_factory=dict)
    reason: str

class EvalRunResult(BaseModel):
    """Result of running `adk eval` as a subprocess. Never raises on target failure."""
    label: str                    # "before" or "after" (or arbitrary identifier)
    command: list[str]
    exit_code: int
    pass_count: Optional[int] = None
    fail_count: Optional[int] = None
    pass_rate: Optional[float] = None
    stdout: str
    stderr: str
    duration_seconds: float

class CoverageReport(BaseModel):
    audit_id: str
    target_agent: str
    corpus_source: str
    corpus_size: int
    noise_fraction: float                          # Items HDBSCAN labeled as noise
    eval_count: int
    coverage_score: float                          # [0, 1]
    regions: list[Region]
    gaps: list[Gap]                                # Sorted by risk descending
    redundant_evals: list[RedundantEval] = Field(default_factory=list)
    generated_cases: list[GeneratedCase] = Field(default_factory=list)
    eval_run_before: Optional[EvalRunResult] = None
    eval_run_after: Optional[EvalRunResult] = None
    decisions: list[Decision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)  # e.g. "noise_fraction high"
    created_at: datetime
```

These schemas are stable. If a sub-agent needs additional fields, extend via the `metadata` / `details` dicts — do not change the schema mid-build.

---

## 7. Core Modules

### 7.1 Embeddings (`cartograph/core/embeddings.py`)

```python
async def embed_texts(texts: list[str], cache_key: str | None = None) -> np.ndarray:
    """
    Returns shape (len(texts), 768) array.
    Uses google.genai client. Batches in groups of 100.
    If cache_key is provided, caches to data/embeddings/{cache_key}.npy
    and returns from cache on subsequent calls.
    """
```

**Decisions:**
- Always cache. Embeddings are expensive and deterministic.
- If a single batch fails, retry once with exponential backoff, then raise.
- Cache key convention: `{source}_{sha256(joined_texts)[:16]}`.

### 7.2 Clustering (`cartograph/core/clustering.py`)

```python
import umap
import hdbscan
import numpy as np

def fit_reducer(embeddings: np.ndarray, target_dim: int = 30) -> umap.UMAP:
    """
    Fits a UMAP reducer on corpus embeddings.
    n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42 (reproducibility).
    Returns the fitted UMAP object — caller is responsible for persisting it
    via joblib so subsequent transforms (evals, generated cases) project into
    the SAME space.
    """

def transform_embeddings(reducer: umap.UMAP, embeddings: np.ndarray) -> np.ndarray:
    """
    Transforms new embeddings using a fitted reducer. Used to project eval cases
    and generated cases into the same 30d space as the corpus and regions.
    Returns shape (N, 30).
    """

def cluster(reduced_30d: np.ndarray, min_cluster_size: int = 20) -> np.ndarray:
    """
    HDBSCAN on the already-reduced 30d embeddings.
    Returns int labels; -1 means noise. min_samples=5, metric='euclidean'
    (we're already in UMAP-reduced space, where Euclidean is appropriate).
    """

def fit_visualization_reducer(embeddings: np.ndarray) -> umap.UMAP:
    """
    Fits a SEPARATE UMAP for 2D visualization.
    n_neighbors=15, min_dist=0.1 (more spread for visualization).
    Visualization reducer is also persisted (reducer_2d.joblib) so the same
    layout can be reused when projecting evals onto the visualization.
    """
```

**Decisions:**
- Two separate UMAP fits: 30d for clustering+regions, 2d for visualization. Don't reuse.
- Both reducers are persisted per audit (`reducer_30d.joblib`, `reducer_2d.joblib`).
- For corpora < 1000 items, drop `min_cluster_size` to 10. Make this a parameter.
- Noise points (label -1) are NOT included in regions. Track them separately.
- **Numerical convention:** all 30d embeddings used for distance computations are L2-normalized after the UMAP transform (each row divided by its L2 norm). With L2-normalized vectors, cosine distance = `1 - dot_product` and lies in `[0, 2]`; we further clamp to `[0, 1]` for any quantity that feeds into the risk score (heterogeneity, distance-to-nearest-eval) so risk score components remain bounded and comparable across regions. Original 768d embeddings used for redundancy checks are NOT normalized — those are passed to Gemini cosine similarity which handles normalization internally.

### 7.3 Coverage Metric (`cartograph/core/coverage.py`)

```python
def compute_coverage(regions: list[Region]) -> float:
    """
    coverage = sum(density_i * covered_i) / sum(density_i)
    where covered_i = 1 if region.eval_case_ids else 0.
    Returns float in [0, 1].
    Pure function over already-populated Region objects.
    """

def compute_noise_fraction(total_count: int, noise_count: int) -> float:
    """noise_count / total_count, clamped to [0, 1]."""

def coverage_warnings(noise_fraction: float, regions: list[Region]) -> list[str]:
    """
    Returns list of human-readable warnings. Currently:
    - if noise_fraction > 0.20: "noise_fraction is high — coverage may be less reliable"
    - if any region.member_count < 5: "very small regions detected"
    - if total regions < 3: "corpus produced too few regions"
    """
```

**Noise handling:** Coverage is computed over clustered corpus mass only. Items labeled as noise by HDBSCAN are excluded from regions and from the denominator. The audit report includes `noise_fraction = noise_count / total_corpus_count`. If `noise_fraction > 0.20`, a warning is added to `CoverageReport.warnings` indicating coverage may be less reliable in this domain. If `noise_fraction > 0.50`, the audit fails with a clear error rather than producing a misleading score.

### 7.4 Risk Score (`cartograph/core/risk.py`)

```python
def compute_risk(region: Region, eval_embeddings_30d: np.ndarray) -> dict:
    """
    risk = density * heterogeneity * distance_to_nearest_eval

    All distances computed in 30d UMAP space using cosine distance.
    region.centroid is 30d. eval_embeddings_30d is shape (N_evals, 30).
    Both are assumed to be L2-normalized (see §7.2 numerical convention).
    distance_to_nearest_eval is clamped to [0, 1] before multiplication so
    risk_score is bounded by density * heterogeneity (each in [0, 1]).

    Computed for uncovered regions only (callers filter beforehand).

    Returns: {
        "risk_score": float,
        "components": {"density": ..., "heterogeneity": ..., "distance": ...}
    }
    """
```

### 7.5 Storage (`cartograph/core/storage.py`)

**Per-audit directory layout:**

```
data/audits/{audit_id}/
  ├── embeddings_corpus_full.npy      (N_corpus x 768)
  ├── embeddings_corpus_30d.npy       (N_corpus x 30)
  ├── embeddings_corpus_2d.npy        (N_corpus x 2, visualization only)
  ├── embeddings_evals_full.npy       (N_evals x 768)
  ├── embeddings_evals_30d.npy        (N_evals x 30, transformed via reducer_30d)
  ├── embeddings_generated_full.npy
  ├── embeddings_generated_30d.npy
  ├── reducer_30d.joblib              (fitted UMAP for clustering space)
  ├── reducer_2d.joblib               (fitted UMAP for visualization)
  ├── cluster_labels.npy              (int array, -1 means noise)
  └── viz/
      ├── regions.png
      ├── coverage.png
      └── before_after.png
```

**SQLite schema** (`data/audits/cartograph.db` — single DB across all audits):

```sql
CREATE TABLE audits (
    id TEXT PRIMARY KEY,
    target_agent TEXT NOT NULL,
    corpus_source TEXT NOT NULL,
    corpus_size INTEGER,
    noise_fraction REAL,
    coverage_score REAL,
    eval_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    warnings_json TEXT             -- list of warning strings
);

CREATE TABLE regions (
    audit_id TEXT REFERENCES audits(id),
    region_id TEXT,
    label TEXT,
    density REAL,
    heterogeneity REAL,
    eval_density REAL,
    member_count INTEGER,
    exemplar_corpus_ids_json TEXT,   -- list[str], top-5 exemplars
    centroid_path TEXT,              -- Path to .npy file with the 30d centroid
    PRIMARY KEY (audit_id, region_id)
);

CREATE TABLE corpus_items (
    audit_id TEXT REFERENCES audits(id),
    item_id TEXT,
    text TEXT,
    source TEXT,
    metadata_json TEXT,
    region_id TEXT,                  -- nullable; null means noise
    PRIMARY KEY (audit_id, item_id)
);

CREATE TABLE eval_cases (
    audit_id TEXT REFERENCES audits(id),
    case_id TEXT,
    source_path TEXT,
    content TEXT,
    raw_json TEXT,
    region_id TEXT,
    assignment_distance REAL,
    PRIMARY KEY (audit_id, case_id)
);

CREATE TABLE generated_cases (
    audit_id TEXT REFERENCES audits(id),
    case_id TEXT,
    region_id TEXT,
    content TEXT,
    raw_json TEXT,
    grounded_in_json TEXT,           -- list[str] of corpus_ids
    validation_status TEXT,
    validation_details_json TEXT,
    accepted BOOLEAN,
    PRIMARY KEY (audit_id, case_id)
);

CREATE TABLE gaps (
    audit_id TEXT REFERENCES audits(id),
    region_id TEXT,
    risk_score REAL,
    components_json TEXT,            -- {"density": ..., "heterogeneity": ..., "distance": ...}
    rank INTEGER,
    PRIMARY KEY (audit_id, region_id)
);

CREATE TABLE redundant_evals (
    audit_id TEXT REFERENCES audits(id),
    case_id TEXT,
    duplicate_of_case_id TEXT,
    similarity REAL,                 -- 768d cosine similarity
    PRIMARY KEY (audit_id, case_id, duplicate_of_case_id)
);

CREATE TABLE eval_runs (
    audit_id TEXT REFERENCES audits(id),
    label TEXT,                      -- "before" or "after"
    exit_code INTEGER,
    pass_count INTEGER,
    fail_count INTEGER,
    pass_rate REAL,
    stdout TEXT,
    stderr TEXT,
    duration_seconds REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (audit_id, label)
);

CREATE TABLE decisions (
    audit_id TEXT REFERENCES audits(id),
    step INTEGER,
    timestamp TIMESTAMP,
    action TEXT,
    inputs_json TEXT,
    outputs_json TEXT,
    reason TEXT,
    PRIMARY KEY (audit_id, step)
);
```

**Storage API (key functions):**

```python
def init_db() -> None
def create_audit(audit_id: str, target_agent: str, corpus_source: str) -> None
def save_corpus_items(audit_id: str, items: list[CorpusItem]) -> None
def save_corpus_embeddings(audit_id: str, full: np.ndarray, reduced_30d: np.ndarray, reduced_2d: np.ndarray) -> None
def save_reducer(audit_id: str, reducer: umap.UMAP, kind: Literal["30d", "2d"]) -> None
def load_reducer(audit_id: str, kind: Literal["30d", "2d"]) -> umap.UMAP
def save_regions(audit_id: str, regions: list[Region]) -> None
def save_eval_cases(audit_id: str, cases: list[EvalCase]) -> None
def save_gaps(audit_id: str, gaps: list[Gap]) -> None
def save_redundant_evals(audit_id: str, redundant: list[RedundantEval]) -> None
def save_generated_cases(audit_id: str, cases: list[GeneratedCase]) -> None
def save_eval_run(audit_id: str, result: EvalRunResult) -> None
def save_decision(audit_id: str, decision: Decision) -> None
def load_coverage_report(audit_id: str) -> CoverageReport
```

`load_coverage_report` reconstructs the full `CoverageReport` from all tables. This is what `coverage_state` MCP server calls.

### 7.6 ADK Eval Runner (`cartograph/core/adk_eval.py`)

```python
import subprocess
import time
from typing import Optional
from .models import EvalRunResult

def run_adk_eval(
    agent_path: str,
    evalset_path: str,
    label: str = "audit",
    timeout_seconds: int = 600
) -> EvalRunResult:
    """
    Runs `adk eval <agent_path> <evalset_path>` as a subprocess.
    Captures stdout, stderr, exit code, and duration.
    Parses pass_count, fail_count, pass_rate when stdout matches expected format.
    If parsing fails, stores raw output and pass_rate=None.

    NEVER raises for target-agent failures. A failed target eval is information,
    not a Cartograph error. The caller decides how to react.

    Times out after timeout_seconds; if so, exit_code=124 and stderr includes
    a clear timeout message.
    """

def parse_adk_output(stdout: str) -> tuple[Optional[int], Optional[int], Optional[float]]:
    """
    Best-effort parser for ADK eval CLI output.

    Looks for patterns in this order:
    1. JSON output (if `adk eval --json` is supported)
    2. "X/Y passed" or "X out of Y" patterns
    3. "pass_rate: 0.85" or "passed: 0.85" patterns

    Returns (pass_count, fail_count, pass_rate); any may be None when not parseable.
    """
```

**Decisions:**
- Subprocess invocation, not Python API import. ADK eval CLI is the contract.
- Never modify the original evalset file. The Generator writes a sibling `<original>_cartograph_generated.evalset.json` and the "after" run uses a temporary merged file written to `data/audits/{audit_id}/eval_run_after.evalset.json`.
- All eval runs are persisted via `save_eval_run` regardless of success/failure.

---

## 8. Sub-Agents

All three are ADK `LlmAgent`s registered as `sub_agents` of the root. They communicate by reading/writing the audit row in SQLite (the shared coverage state). Each takes an `audit_id` as the primary input via session state.

**Escape valve for ADK API drift.** If the exact ADK sub-agent registration API differs from `LlmAgent(sub_agents=[...])` (the SDK is pre-1.0 and may evolve), preserve the architectural shape: a root agent that delegates to three named sub-agents (Mapper, Auditor, Generator), where each sub-agent is itself an ADK agent (not a plain Python function) and produces decision-log entries on invocation. Equivalent ADK constructs (`SequentialAgent` chains, `Workflow` patterns, agent-as-tool) are acceptable substitutes. Raw Python function dispatching disguised as agents is NOT acceptable — the demo must show distinct agent steps in the decision log, and the orchestrator must be making LLM-driven decisions about which sub-agent to invoke.

### 8.1 Mapper (`cartograph/agents/mapper.py`)

**System prompt** (verbatim):
> You are the Mapper sub-agent of Cartograph. Your job: given an audit_id, load the corpus from storage (via the corpus_reader MCP server), embed it with Gemini, fit a UMAP reducer to 30d, persist the reducer, cluster with HDBSCAN, label each region with a short human-readable description from its top exemplars, save all of this to storage. Then return a one-paragraph summary: number of regions found, density distribution, noise fraction.

**Tool functions exposed to the LLM:**
- `load_corpus(audit_id: str, corpus_source: str, query_args: dict) -> int` — calls corpus_reader MCP server, persists items, returns count
- `compute_embeddings(audit_id: str) -> dict` — embeds all corpus items, returns {duration_s, count, cached}
- `fit_and_persist_reducer(audit_id: str) -> dict` — fits UMAP 30d reducer on corpus, transforms corpus to 30d, L2-normalizes, persists reducer + reduced array
- `run_clustering(audit_id: str, min_cluster_size: int = 20) -> dict` — runs HDBSCAN on 30d, returns region count + noise fraction
- `label_regions(audit_id: str) -> list[dict]` — uses Gemini 2.5 Flash to produce labels from top-5 exemplars per region (cached)
- `save_regions(audit_id: str) -> None` — persists Region objects to SQLite

**Density formula (explicit):** for each region, `density = member_count / total_clustered_count` where `total_clustered_count = total_corpus_count - noise_count`. The denominator is non-noise corpus only — a region holding 30% of *clustered* items has density 0.3 even if 40% of total corpus was labeled noise. This makes coverage scores comparable across audits with different noise fractions.

**Definition of Done:**
- [ ] Given Bitext corpus of 1000+ items, produces ≥6 regions with non-trivial density distribution
- [ ] Noise fraction is logged and persisted (`audits.noise_fraction`)
- [ ] Labels are human-readable, not just `"region_03"`
- [ ] `reducer_30d.joblib` exists and can be loaded with `joblib.load` to produce the same 30d output as in-process transformation
- [ ] Total time < 90s on a standard laptop after embedding cache is warm
- [ ] Idempotent: running twice on the same audit_id produces identical results (same labels, same region count)

### 8.2 Auditor (`cartograph/agents/auditor.py`)

**System prompt:**
> You are the Auditor sub-agent of Cartograph. Your job: given an audit_id with regions already populated, load the eval suite (via eval_io MCP server), embed each eval case, project it into 30d using the persisted reducer, assign each eval to its nearest region centroid (cosine distance), compute the coverage score, identify uncovered regions, rank gaps by risk score, flag redundant evals (cosine similarity > 0.85 in 768d space), and persist all results. Return a one-paragraph summary: coverage score, region counts (covered/uncovered), top 3 gaps by risk.

**Tool functions:**
- `load_eval_suite(audit_id: str, eval_path: str) -> int` — calls eval_io_adk MCP server
- `compute_eval_embeddings(audit_id: str) -> dict` — embeds eval contents (768d) and transforms to 30d via persisted reducer
- `assign_evals_to_regions(audit_id: str) -> dict` — for each eval, find nearest region centroid in 30d, log assignment_distance, return count of suspicious assignments (distance > 0.35)
- `compute_coverage(audit_id: str) -> float` — uses `core.coverage.compute_coverage`
- `compute_gaps(audit_id: str) -> list[dict]` — for uncovered regions, computes risk and persists Gap rows
- `flag_redundancies(audit_id: str, threshold: float = 0.85) -> list[dict]` — pairwise cosine on 768d eval embeddings
- `save_audit_results(audit_id: str) -> None`

**Eval-to-region assignment rule:** Each eval is assigned to its nearest region centroid in 30d cosine distance. If `assignment_distance > 0.35`, the eval is still assigned (we always pick a region) but flagged as "suspicious" in `CoverageReport.warnings`. This is a v1 simplification; a stricter no-assignment threshold is on the v2 roadmap.

**Definition of Done:**
- [ ] Coverage score is in [0, 1]
- [ ] Gaps are sorted by risk descending and persisted to `gaps` table
- [ ] Redundancies use the documented threshold and persisted to `redundant_evals` table
- [ ] All evals have `region_id` and `assignment_distance` populated
- [ ] All region-level distance computations verified to use 30d (unit test asserts dimensions)

### 8.3 Generator (`cartograph/agents/generator.py`)

**System prompt:**
> You are the Generator sub-agent of Cartograph. Your job: given an audit_id and a list of target region_ids, generate new eval cases that target those regions. For each region: retrieve the top exemplars, prompt Gemini 2.5 Pro to produce N candidate eval cases grounded in those exemplars and matching the target eval format, validate each candidate by embedding it (768d AND 30d) and applying two filters, persist accepted candidates with provenance. To guarantee the redundancy filter is exercised, include the nearest existing eval as a deliberate negative-control candidate among the inputs.

**Validation filters** (applied in this order):

1. **Redundancy** (768d cosine similarity to existing evals):
   - If max similarity to any existing eval > 0.85 → `rejected_redundant`
2. **Off-corpus** (30d cosine distance to region exemplars):
   - If min cosine **distance** from candidate to any region exemplar > 0.42 → `rejected_off_corpus`
   - (Even the closest exemplar is too far → candidate has drifted off-distribution)
3. Otherwise → `accepted`

**Generation prompt template** (use exactly):

```
You are generating new test cases for an agent eval suite.

The agent operates in this domain:
{region_label}

Here are 5 representative real user inputs from this domain:
1. {exemplar_1}
2. {exemplar_2}
3. {exemplar_3}
4. {exemplar_4}
5. {exemplar_5}

Generate {n} new test inputs that:
- Are similar in style and complexity to the exemplars
- Cover variations the exemplars hint at but don't all show
- Are realistic things a user might actually say

Return as a JSON array. Each item: {"input": "...", "expected_intent_or_behavior": "..."}
```

**Tool functions:**
- `generate_for_region(audit_id: str, region_id: str, n: int = 6) -> list[GeneratedCase]`
- `validate_candidates(audit_id: str, candidates: list[GeneratedCase]) -> list[GeneratedCase]` — runs both filters, populates `validation_status`
- `format_for_target(audit_id: str, accepted: list[GeneratedCase], format: Literal["adk_evalset", "yaml", "jsonl"]) -> str` — calls eval_io_adk.write_evalset
- `save_generated(audit_id: str) -> None`

**Definition of Done:**
- [ ] Each accepted case has at least 3 corpus exemplars in `grounded_in_corpus_ids`
- [ ] In tests, `mock_gemini` returns one deliberately duplicate candidate (cosine ~0.95 to existing eval) and one deliberately off-corpus candidate (random embedding); validation MUST reject both. This is the deterministic acceptance test for the validation logic.
- [ ] In the live demo, Generator deliberately includes the nearest existing eval as a negative-control candidate so the redundancy-rejection path is exercised every demo run regardless of what Gemini produces
- [ ] Format conversion produces valid `.evalset.json` for ADK target (round-trips through `eval_io_adk.read_evalset`)

---

## 9. Root Orchestrator (`cartograph/agents/root.py`)

ADK `LlmAgent` with Mapper, Auditor, Generator declared as `sub_agents`. The root is the agent that judges interact with.

**System prompt** (verbatim):

```
You are Cartograph, an autonomous agent that audits eval suites.

You have three sub-agents available:
- mapper: maps the real-world corpus into semantic regions
- auditor: scores coverage and ranks gaps
- generator: produces new eval cases for uncovered regions

You also have these decision functions:
- decide_generate_target(audit_id) -> dict
- decide_terminate(audit_id) -> bool

Your workflow for any audit:

1. Call transfer_to_agent("mapper") to map the corpus.
2. Call transfer_to_agent("auditor") to score coverage.
3. If coverage >= COVERAGE_THRESHOLD (default 0.75), call decide_terminate. Skip to step 7.
4. Call decide_generate_target to choose top-K uncovered regions to target.
5. Call transfer_to_agent("generator") with those region_ids.
6. Call transfer_to_agent("auditor") to re-audit. Then return to step 3.
7. Generate the final report using format_final_report.

At every transition, log a Decision row with the action, inputs, outputs, and your one-line reason.

Loop limit: max 3 generation rounds total. After 3 rounds, terminate regardless of coverage.

Be decisive. Do not ask for clarification. Do not propose alternatives. Execute the workflow.
```

**Tool functions on the root:**
- `decide_generate_target(audit_id: str, k: int = 2) -> dict` — returns top-K regions by risk
- `decide_terminate(audit_id: str, threshold: float = 0.75) -> dict` — returns {"terminate": bool, "reason": str}
- `format_final_report(audit_id: str) -> CoverageReport`
- `log_decision(audit_id: str, action: str, inputs: dict, outputs: dict, reason: str) -> None`

**Definition of Done:**
- [ ] On a synthetic high-coverage fixture (coverage > 0.8), Generator is never invoked
- [ ] On customer-service audit, Generator is invoked exactly once or twice
- [ ] Decision log has at least 7 rows for the customer-service audit
- [ ] Loop limit enforced; pathological cases exit cleanly

---

## 10. MCP Servers

All servers use `mcp` Python SDK with stdio transport. Each is a standalone runnable: `python -m cartograph.mcp_servers.corpus_reader_bitext`.

Common scaffolding lives in `cartograph/mcp_servers/_common.py`.

### 10.1 `corpus_reader_bitext`

**Tools:**
- `list_intents() -> list[str]`
- `fetch_examples(intent: str | None = None, limit: int = 1000, offset: int = 0) -> list[dict]`
  - Returns `[{id, text, intent, category}]`
- `get_metadata() -> dict`

**Implementation:**
- Loads `bitext/Bitext-customer-support-llm-chatbot-training-dataset` from HuggingFace
- Caches to `data/corpora/bitext.parquet` after first download
- All tools read from the cache

### 10.2 `corpus_reader_stackexchange`

**Tools:**
- `fetch_questions(site: str, tag: str | None = None, limit: int = 500) -> list[dict]`
  - `site` is the Cartograph short name (see map below)
  - Returns `[{id, text (= title + body excerpt), tags, score}]`

**Site mapping** (Cartograph short name → `stackapi` site identifier):

```python
STACKEXCHANGE_SITE_MAP = {
    "stackoverflow": "stackoverflow",
    "money":         "money",
    "travel":        "travel",
    "datascience":   "datascience",
}
```

The `stackapi` library uses the subdomain prefix of `*.stackexchange.com` (or `stackoverflow` for the main site) as its site identifier. The map exists so callers in Cartograph use one canonical short name; the MCP server resolves it to whatever `stackapi` expects. If a future `stackapi` version changes its identifiers, fix the map only — callers don't change.

Uses `stackapi`. Rate-limit aware. Cache to `data/corpora/se_{site}_{tag_or_none}.json`.

### 10.3 `corpus_reader_github`

**Tools:**
- `fetch_issues(owner: str, repo: str, limit: int = 500, state: str = "closed") -> list[dict]`
  - Returns `[{id, text (= title + body), labels}]`

Uses `PyGithub`. Requires `GITHUB_TOKEN` env var.

### 10.4 `eval_io_adk`

**Tools:**
- `read_evalset(path: str) -> list[dict]` — reads `.evalset.json`, returns list of normalized eval cases
- `write_evalset(path: str, cases: list[dict], merge: bool = True) -> dict` — writes back; `merge=True` appends to existing, `merge=False` replaces

**Supported ADK Golden Dataset format (v1):**

```json
{
  "eval_set_id": "customer_service_v1",
  "name": "Customer Service Eval Set",
  "eval_cases": [
    {
      "eval_id": "refund_simple_001",
      "conversation": [
        {
          "role": "user",
          "content": "I want a refund for order #12345"
        }
      ],
      "expected_tool_use": [
        {
          "tool_name": "lookup_order",
          "arguments": {"order_id": "12345"}
        }
      ],
      "expected_intermediate_responses": [],
      "expected_final_response": "I've processed your refund for order #12345.",
      "session_input": {}
    }
  ]
}
```

**Normalization:** convert to `{id, content, raw}` shape where:
- `id` = `eval_id`
- `content` = concatenation of all `conversation[].content` entries with role `user`, joined by `\n`
- `raw` = the entire original eval_case object, preserved verbatim

**Round-trip rule:** if the real ADK evalset contains fields not enumerated above (e.g., new fields added in a future ADK version), preserve them inside `raw` and round-trip them unchanged when writing back. Cartograph never strips unknown fields.

### 10.5 `eval_io_git`

Generic file-based eval I/O for non-ADK formats.

**Tools:**
- `read_yaml(path: str) -> list[dict]`
- `read_jsonl(path: str) -> list[dict]`
- `write_yaml(path: str, cases: list[dict]) -> dict`
- `write_jsonl(path: str, cases: list[dict]) -> dict`

### 10.6 `coverage_state` (the canonical provider)

**Tools:**
- `list_audits(limit: int = 20) -> list[dict]`
- `get_coverage(audit_id: str) -> dict` — full CoverageReport as JSON
- `get_uncovered_regions(audit_id: str, top_k: int = 5) -> list[dict]`
- `get_decision_log(audit_id: str) -> list[dict]`
- `get_generated_cases(audit_id: str, accepted_only: bool = True) -> list[dict]`

This is the only **provider** server; the other five are client adapters Cartograph itself consumes.

### 10.7 MCP Boundary Map

When does a sub-agent talk to MCP, and when does it call internal Python? The rule:

- **MCP boundary:** anything that crosses the product's edge — reading external corpora, reading/writing eval suites, exposing coverage state to outside systems.
- **Direct Python:** anything internal to a single audit — storage I/O, embeddings via Gemini SDK, clustering, validation math, decision logging, ADK eval subprocess invocation.

Concretely:

| Operation | Implementation |
|---|---|
| Mapper.load_corpus | Calls `corpus_reader_*` MCP server (bitext, github, stackexchange) |
| Mapper.compute_embeddings | Direct (Gemini SDK in `core/embeddings.py`) |
| Mapper.fit_and_persist_reducer | Direct (UMAP via `core/clustering.py`) |
| Mapper.run_clustering | Direct (HDBSCAN via `core/clustering.py`) |
| Mapper.save_regions | Direct (SQLite via `core/storage.py`) |
| Auditor.load_eval_suite | Calls `eval_io_adk` MCP server |
| Auditor.compute_eval_embeddings | Direct + uses persisted reducer |
| Auditor.assign_evals_to_regions | Direct (NumPy cosine in 30d) |
| Auditor.compute_coverage / gaps / redundancies | Direct (`core/coverage.py`, `core/risk.py`) |
| Generator.generate_for_region | Direct (Gemini 2.5 Pro call) |
| Generator.validate_candidates | Direct (`core/validation.py`) |
| Generator.format_for_target | Calls `eval_io_adk.write_evalset` MCP tool |
| Root.run_adk_eval (before/after) | Direct subprocess (`core/adk_eval.py`) |
| `coverage_state` server | Provider — exposes audit results to external systems via MCP |

The pattern: every external boundary is MCP. Every internal computation is direct Python. This keeps the agent's surface composable (any new corpus or eval store is a new MCP adapter, not a code change) without paying the protocol overhead for in-process operations.

---

## 11. CLI (`cartograph/cli.py`)

Typer app with these commands:

```bash
# Run an audit end-to-end
cartograph audit \
    --target adk-samples/customer-service \
    --corpus bitext \
    --eval-path path/to/customer_service.evalset.json \
    [--coverage-threshold 0.75] \
    [--corpus-limit 5000]

# Show a report from a previous audit
cartograph report <audit_id>
cartograph report --latest

# Demo commands
cartograph demo prepare              # Precompute fleet benchmark for all 5 agents
cartograph demo run customer-service # Live deep-dive
cartograph demo visualize <audit_id> # Generate 2D map images

# Utility
cartograph corpus list               # Show available corpus_reader adapters
cartograph version
```

The `audit` command prints the orchestrator decision log live to terminal (using `rich`). Every Decision row is printed as it's written.

---

## 12. Demo Infrastructure

### 12.1 Precomputation (`cartograph/demo/prepare.py`)

Single function: `prepare_fleet_benchmark()`. Runs `cartograph audit` against all 5 agents non-interactively, saves results to `data/precomputed/`. Idempotent. Skips agents already done.

The 5 agents and their corpora:

| Target agent (path in adk-samples) | Corpus reader | Corpus query |
|---|---|---|
| `python/agents/customer-service` | bitext | full dataset |
| `python/agents/software-bug-assistant` | github | `psf/requests` issues |
| `python/agents/financial-advisor` | stackexchange | `money`, top 500 |
| `python/agents/travel-concierge` | stackexchange | `travel`, top 500 |
| `python/agents/data-science` | stackexchange | `stackoverflow` tag `pandas`, top 500 |

**Real vs mocked corpora.** External APIs (StackExchange, GitHub) can rate-limit, return errors, or be unreachable on demo day. The fleet benchmark must produce a JSON artifact regardless. Therefore, each per-agent result records `corpus_mode`:

- `"real"` — the corpus was successfully fetched from the live source (or its on-disk cache).
- `"mocked"` — the live source failed AND no cache existed; a 30-item fallback fixture from `cartograph/demo/mocks/{corpus_reader}.json` was used. Mocked entries also record `corpus_source_failure_reason` (free text).

The fallback ladder is: live source → on-disk cache → bundled mock fixture. The first one that succeeds wins. Mock fixtures are checked into the repo as part of the package (NOT under `data/`, which is gitignored), so fallback is deterministic regardless of network state.

`fleet_benchmark.json` shape:

```json
{
  "generated_at": "2026-05-15T12:34:56Z",
  "agents": [
    {
      "target_agent": "python/agents/customer-service",
      "corpus_mode": "real",
      "corpus_size": 27113,
      "noise_fraction": 0.08,
      "coverage_score": 0.41,
      "region_count": 11,
      "audit_id": "..."
    },
    {
      "target_agent": "python/agents/travel-concierge",
      "corpus_mode": "mocked",
      "corpus_source_failure_reason": "stackexchange API rate limit",
      "corpus_size": 30,
      "noise_fraction": 0.03,
      "coverage_score": 0.55,
      "region_count": 4,
      "audit_id": "..."
    }
  ]
}
```

**Definition of Done:**
- [ ] All 5 agents have an entry in `fleet_benchmark.json` with non-null coverage_score and region_count
- [ ] At least the `customer-service` entry has `corpus_mode == "real"` (this is non-negotiable — see §13.1)
- [ ] Each mocked entry includes a non-empty `corpus_source_failure_reason`
- [ ] Total runtime under 30 minutes with cold caches and all real corpora succeeding
- [ ] Re-running uses caches and completes in under 5 minutes
- [ ] `data/precomputed/fleet_benchmark.json` exists and is valid JSON

### 12.2 Live Deep-Dive (`cartograph/demo/deep_dive.py`)

Runs the customer-service audit live with these constraints:
- Embeddings already cached (from prepare step)
- Corpus already cached
- Mapper output already in storage (just reloaded)
- Auditor runs live (fast: just projection + scoring)
- Generator runs live for top 2 regions, ~6 candidates each (with deliberate negative-control candidate)
- Re-audit runs live
- `adk eval` runs live before and after
- Final report rendered live

Total target runtime: under 90 seconds. Use `rich` Live display to show decision log streaming.

**Definition of Done:**
- [ ] `data/precomputed/customer_service_report.json` exists after run
- [ ] stdout includes both `pass_rate_before` and `pass_rate_after` lines
- [ ] stdout includes `coverage_before` and `coverage_after` lines
- [ ] Visible "rejection" event for at least one Generator candidate

### 12.3 Visualization (`cartograph/demo/visualize.py`)

Produces these PNGs in `data/audits/{audit_id}/viz/`:
- `regions.png` — 2D scatter of corpus, colored by region
- `coverage.png` — same scatter, eval cases overlaid as larger markers; uncovered regions tinted red
- `before_after.png` — two-panel side-by-side: before-generation and after-generation coverage maps

Uses `matplotlib`. Style: clean, publication-ready, no chart junk. 200 DPI.

**Definition of Done:**
- [ ] All three PNGs generated for customer-service audit
- [ ] Visible legend showing region labels
- [ ] Eval cases distinguishable from corpus points (larger marker, different shape)
- [ ] Red tint clearly visible on uncovered regions

---

## 13. Definition of Done — Master Checklist

The project is shippable when ALL of these are checked:

**Core**
- [ ] `pip install -e .` succeeds in a clean Python 3.11 env
- [ ] `cartograph version` prints version
- [ ] `pytest` passes (target: 30+ tests, all green)
- [ ] `mypy cartograph/core` passes in strict mode
- [ ] `ruff check cartograph` passes
- [ ] All region-level distance computations verified to use 30d (unit tests assert array dims)
- [ ] Persisted UMAP reducers round-trip correctly: `load_reducer(...).transform(x)` produces same output as in-process `reducer.transform(x)`

**MCP servers** (each must run standalone via `python -m`)
- [ ] `corpus_reader_bitext` returns at least 100 items via `fetch_examples`
- [ ] `corpus_reader_stackexchange` returns at least 50 items for `money` site
- [ ] `corpus_reader_github` returns at least 20 issues from `psf/requests`
- [ ] `eval_io_adk` reads a real adk-samples `.evalset.json` and writes a modified version that round-trips through `read_evalset` unchanged for known fields, with unknown fields preserved
- [ ] `eval_io_git` round-trips YAML and JSONL
- [ ] `coverage_state` returns valid JSON for all tools after an audit completes

**Sub-agents**
- [ ] Mapper produces ≥6 regions on Bitext (5000 items)
- [ ] Mapper persists `reducer_30d.joblib` and `reducer_2d.joblib`
- [ ] Auditor coverage score is reproducible (same inputs → same score)
- [ ] Auditor populates `assignment_distance` for every eval
- [ ] Generator validation rejects both the deliberate-redundant and deliberate-off-corpus candidates in mocked tests (deterministic)

**Orchestrator**
- [ ] On a synthetic high-coverage fixture, Generator is skipped and orchestrator terminates
- [ ] On customer-service, Generator is invoked exactly once or twice
- [ ] Decision log has ≥7 rows for the customer-service audit
- [ ] Loop limit is enforced

**Demo**
- [ ] `cartograph demo prepare` completes for all 5 agents in <30 min cold, <5 min warm
- [ ] `cartograph demo run customer-service` completes in <90s with warm caches
- [ ] All visualization PNGs render cleanly
- [ ] Running the demo shows: orchestrator log streaming, before/after pass-rate comparison, generated cases linked to exemplars

**Documentation**
- [ ] README has: install instructions, env vars needed, quickstart command, demo command
- [ ] README links to the proposal markdown
- [ ] Each MCP server module has a docstring describing its tools

**Submission readiness**
- [ ] A 4-minute screen recording of the demo flow exists
- [ ] The demo recording matches the demo arc in the proposal §9
- [ ] Repository is public on GitHub, devpost link prepared

### 13.1 Final Acceptance: `scripts/run_demo.sh`

The single command `bash scripts/run_demo.sh` is the project's final acceptance test. It runs the fleet prepare + customer-service deep-dive + visualization end-to-end. Two acceptance modes exist:

**Full acceptance** (preferred for submission):
- All 5 agents in `fleet_benchmark.json` have `corpus_mode == "real"`
- Live external APIs (StackExchange, GitHub, HuggingFace) succeeded

**Fallback acceptance** (acceptable for submission if external APIs fail):
- `customer-service` MUST have `corpus_mode == "real"` (Bitext is local after first download — no excuse for it being mocked)
- The other 4 agents MAY be `"mocked"` with `corpus_source_failure_reason` populated
- The recording narration acknowledges fallback mode if any agent is mocked

Either mode satisfies acceptance. The script exits 0 in both cases. The only failure case is customer-service being mocked, or any of the file/string conditions below failing.

**Files that must exist after a successful run:**
- `data/precomputed/fleet_benchmark.json`
- `data/precomputed/customer_service_report.json`
- `data/audits/<latest_audit_id>/viz/regions.png`
- `data/audits/<latest_audit_id>/viz/coverage.png`
- `data/audits/<latest_audit_id>/viz/before_after.png`
- `data/audits/<latest_audit_id>/reducer_30d.joblib`
- `data/audits/<latest_audit_id>/reducer_2d.joblib`

**Strings that must appear in stdout:**
- `coverage_before:`
- `coverage_after:`
- `generated_cases:`
- `decision_log`
- `pass_rate_before:`
- `pass_rate_after:`

**Additional check on `fleet_benchmark.json`:**
- The `customer-service` entry has `corpus_mode == "real"` (asserted by the script; failure exits non-zero with a clear message)

**Exit code:** 0

**Total wall time:** under 5 minutes with warm caches (Bitext downloaded, embeddings cached, `GOOGLE_API_KEY` set).

If any of these conditions fails, the project is not done. The script must fail loudly (non-zero exit, clear error pointing to the failing condition) rather than continue silently. The script is the contract: when it passes, ship.

---

## 14. Out of Scope (Explicit Non-Goals for v1)

These are good ideas. They are NOT in v1. Do not implement them.

- Vertex AI Vector Search (NumPy is sufficient)
- FAISS (deferred until corpus > 1M items)
- Cloud Run deployment
- Authentication, multi-tenancy, RBAC
- Real-time CI integration
- L2/L3/L4 Coverage Stack layers
- Auto-PR creation on the target repo
- Web UI / dashboard
- Streaming embeddings during clustering
- Multi-language support beyond English corpora
- Custom embedding models
- Human-in-the-loop review interface
- Eval-suite quality metrics beyond coverage
- Behavioral judge model (that's L2)
- Trajectory analysis (that's L3)
- Retraining or fine-tuning of any model
- Real-time monitoring of coverage drift
- Cost tracking, usage metering
- Continuous corpus refresh
- Stricter eval-to-region assignment threshold (v1 always assigns to nearest)

If a feature is not explicitly listed in §2 or §3, default to "out of scope."

---

## 15. Timeline (2 Weeks)

Week 1 (~10–15h, evenings/weekends):
- **D1–2:** Repo setup, dependencies, models, storage. Steps 1–3.
- **D3–4:** Embeddings, clustering (with reducer persistence), coverage, risk, ADK eval runner. Steps 4–7.
- **D5–7:** First two MCP servers + Mapper sub-agent end-to-end on Bitext. Steps 8–10.

Week 2 (full-time):
- **D8:** Auditor sub-agent (with eval projection via persisted reducer). Step 11.
- **D9:** Generator sub-agent + validation (with deterministic test fixtures). Step 12.
- **D10:** coverage_state MCP server + Root orchestrator. Steps 13–14.
- **D11:** CLI, other corpus readers, eval_io_git. Steps 15–17.
- **D12:** Demo precomputation pipeline. Step 18.
- **D13:** Live deep-dive script + visualization + integration test. Steps 19–20.
- **D14:** `scripts/run_demo.sh`, README, demo recording, submit. Step 21.

If Week 1 slips, the fallback ladder (§12.1) means external-API corpus readers can be left as stubs that immediately return failure, letting the bundled mock fixtures serve as data sources. The fleet benchmark records `corpus_mode == "mocked"` for the affected agents but still produces a valid `fleet_benchmark.json`. The deep-dive on customer-service still works fully because Bitext is local after first download. Submission acceptance is satisfied as long as customer-service is `corpus_mode == "real"` (see §13.1 Fallback acceptance).

---

## 16. Testing Strategy

Three levels:

**Unit tests** (`tests/test_core_*.py`): Pure-Python core modules. No external services. Use synthetic fixtures.
- `test_core_coverage`: hand-built Region lists with known coverage scores
- `test_core_clustering`: synthetic 2D embeddings with known cluster structure; verifies reducer round-trips
- `test_core_risk`: hand-built regions with known density/heterogeneity/distance
- `test_core_validation`: hand-built embeddings with known cosine similarities and distances; both rejection paths exercised
- `test_core_adk_eval`: subprocess invocation mocked, tests output parsing

**Integration tests** (`tests/test_mcp_*.py`, `tests/test_agent_*.py`): Real MCP servers and sub-agents, but with mocked Gemini API responses (use `pytest-mock`). Tiny corpora (10-50 items) and tiny eval suites (3-5 cases).

The Generator integration test uses `mock_gemini` configured to return:
- 1 candidate that is deliberately a near-duplicate of an existing eval (cosine ~0.95 in 768d) → MUST be rejected as `rejected_redundant`
- 1 candidate with a deliberately random embedding far from all corpus exemplars → MUST be rejected as `rejected_off_corpus`
- 4 candidates that pass both filters → MUST be accepted

This is the deterministic backbone of the validation acceptance test.

**End-to-end smoke test** (`tests/test_e2e_smoke.py`): Real Bitext (limited to 200 items), real Gemini API, full audit pipeline. Tagged `@pytest.mark.e2e`, skipped by default, run before submission.

**Test fixtures** (`tests/conftest.py`):
- `tiny_corpus`: 30 hand-crafted CorpusItems across 3 obvious clusters
- `tiny_eval_suite`: 4 EvalCases that cover only 1 of the 3 clusters
- `mock_gemini`: deterministic embeddings + canned generation responses with the controlled candidates above

The `tiny_corpus` + `tiny_eval_suite` + `mock_gemini` fixture set is THE backbone of testing. Build it first; reuse everywhere.

---

## 17. Common Pitfalls and Decisions Already Made

This section pre-resolves the questions an AI assistant might otherwise ask.

**"How do evals get into the same space as regions?"**
The Mapper persists the fitted UMAP reducer to `data/audits/{audit_id}/reducer_30d.joblib`. The Auditor loads this reducer and calls `transform_embeddings(reducer, eval_embeddings_768d)` to produce 30d eval embeddings. All distance math after this point is in 30d. Never compute a distance between a 768d vector and a 30d vector — type-check this if you can.

**"Why did you pin umap-learn to 0.5.5?"**
joblib serialization compatibility across UMAP versions is fragile. Pinning prevents "audit ran fine yesterday, now reducer fails to load" issues. v1 explicitly trades flexibility for reliability here.

**"Should I use async everywhere?"**
No. ADK requires async at the agent boundary; embeddings batch calls are async. Everything else (clustering, coverage math, storage) is sync.

**"What if Gemini returns malformed JSON in Generator?"**
Retry once with a stricter prompt. If still malformed, log and skip that batch. Don't crash the audit.

**"Should regions without exemplars be allowed?"**
No. After clustering, if any region has fewer than 3 members, merge it into noise. Filter before persisting.

**"What's the minimum corpus size?"**
50 items absolute minimum. Below that, return a friendly error.

**"How do I label regions?"**
Send the top-5 exemplars to Gemini 2.5 Flash with: "Summarize what these 5 inputs have in common in 3-7 words. Return only the label, no commentary." Cache labels per region.

**"What if the eval suite is empty?"**
Coverage = 0, every region is a gap, all gaps tied on density-only ranking (heterogeneity and distance still compute). Generator runs on top 2 by density. Document this case in tests.

**"What if all evals project to the same region?"**
Coverage = density of that region. Other regions are gaps. Normal flow.

**"Should I parallelize sub-agent invocations?"**
No. Mapper → Auditor → (optional Generator → Auditor) is strictly sequential. Don't even think about parallelism in v1.

**"What format do I write generated cases in?"**
ADK `.evalset.json` for ADK targets (default). Generic JSONL otherwise. Detected from the source eval suite's filename.

**"Should I commit generated cases automatically?"**
No. Write to a sibling file `<original>_cartograph_generated.evalset.json` and surface for review. Never modify the original eval file.

**"How do I run `adk eval` for the after run without modifying the target's eval suite?"**
Create a temporary merged file at `data/audits/{audit_id}/eval_run_after.evalset.json` containing original + generated cases, and pass that path to `adk eval`. Never touch the original file.

**"What if `adk eval` fails on the target agent?"**
Capture the failure in `EvalRunResult` (with non-zero exit_code, raw stderr), persist via `save_eval_run`, continue. The "after" pass rate may be None if parsing fails — the report includes raw stdout/stderr in that case so a human can inspect.

**"What if HDBSCAN says everything is noise?"**
If `noise_fraction > 0.5`, the audit fails with a clear error message: "corpus too dispersed for clustering at this min_cluster_size." Log the failure and exit. Do not produce a coverage report. If `0.20 < noise_fraction <= 0.50`, add a warning but continue.

**"Where do API keys go?"**
`.env` file at repo root, loaded via `python-dotenv`. Never logged. Never committed. `.env.example` ships with the keys listed but no values.

**"Should I add telemetry?"**
No telemetry, no analytics, no error reporting service. Hackathon scope.

**"What about Windows support?"**
Linux + macOS only. Document this in README.

**"What if the user asks for a feature not in this spec?"**
Add it to a `FUTURE.md` file. Do not implement it in v1.

---

## End of Spec

If you reach a question this document does not answer, default to whatever choice produces the simplest, most demoable v1. When in doubt, smaller scope wins.
