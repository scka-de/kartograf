from __future__ import annotations

import asyncio
from collections import Counter, defaultdict

import numpy as np

from cartograph.core import clustering, storage
from cartograph.core.coverage import compute_noise_fraction
from cartograph.core.embeddings import embed_texts
from cartograph.core.math import cosine_distance, l2_normalize, stable_cache_key, stable_id
from cartograph.core.models import CorpusItem, Region
from cartograph.mcp_servers import (
    corpus_reader_bitext,
    corpus_reader_github,
    corpus_reader_stackexchange,
)

from .common import log_decision

MAPPER_PROMPT = (
    "You are the Mapper sub-agent of Cartograph. Your job: given an audit_id, load the "
    "corpus from storage (via the corpus_reader MCP server), embed it with Gemini, fit a "
    "UMAP reducer to 30d, persist the reducer, cluster with HDBSCAN, label each region "
    "with a short human-readable description from its top exemplars, save all of this to "
    "storage. Then return a one-paragraph summary: number of regions found, density "
    "distribution, noise fraction."
)


async def run_mapper(
    audit_id: str,
    corpus_source: str,
    corpus_limit: int = 1000,
    query_args: dict | None = None,
) -> dict:
    query_args = query_args or {}
    items = load_corpus(audit_id, corpus_source, corpus_limit, query_args)
    full = await embed_texts(
        [item.text for item in items], stable_cache_key(corpus_source, [i.text for i in items])
    )
    reducer_30d = clustering.fit_reducer(full, 30)
    reduced_30d = clustering.transform_embeddings(reducer_30d, full)
    reducer_2d = clustering.fit_visualization_reducer(full)
    reduced_2d = clustering.transform_embeddings(reducer_2d, full)
    labels = clustering.cluster(reduced_30d, min_cluster_size=20)
    regions = build_regions(items, reduced_30d, labels)
    noise_count = int((labels == -1).sum())
    noise_fraction = compute_noise_fraction(len(items), noise_count)
    if noise_fraction > 0.50:
        raise RuntimeError("corpus too dispersed for clustering at this min_cluster_size")

    for index, item in enumerate(items):
        label = int(labels[index])
        item.embedding = full[index].tolist()
        item.embedding_30d = reduced_30d[index].tolist()
        item.region_id = None if label == -1 else f"region_{label:02d}"

    storage.save_corpus_items(audit_id, items)
    storage.save_corpus_embeddings(audit_id, full, reduced_30d, reduced_2d)
    storage.save_reducer(audit_id, reducer_30d, "30d")
    storage.save_reducer(audit_id, reducer_2d, "2d")
    storage.save_cluster_labels(audit_id, labels)
    storage.save_regions(audit_id, regions)
    output = {
        "corpus_count": len(items),
        "region_count": len(regions),
        "noise_fraction": noise_fraction,
        "density": [region.density for region in regions],
    }
    log_decision(
        audit_id,
        "invoke_mapper",
        {"corpus_source": corpus_source},
        output,
        "mapped corpus into semantic regions",
    )
    return output


def load_corpus(
    audit_id: str,
    corpus_source: str,
    limit: int,
    query_args: dict,
) -> list[CorpusItem]:
    if corpus_source == "bitext":
        rows = corpus_reader_bitext.fetch_examples(limit=limit)
        source = "bitext"
    elif corpus_source == "github":
        rows = corpus_reader_github.fetch_issues(
            query_args.get("owner", "psf"),
            query_args.get("repo", "requests"),
            limit=limit,
        )
        source = "github"
    elif corpus_source == "stackexchange":
        rows = corpus_reader_stackexchange.fetch_questions(
            query_args.get("site", "money"),
            query_args.get("tag"),
            limit=limit,
        )
        source = "stackexchange"
    else:
        raise ValueError(f"unsupported corpus_source: {corpus_source}")
    if not rows:
        raise RuntimeError(f"corpus_source {corpus_source!r} returned no rows")
    if len(rows) < 3:
        expanded = rows * (3 // max(1, len(rows)) + 1)
        rows = expanded[:3]
    items = [
        CorpusItem(
            id=str(row.get("id") or stable_id(source, row.get("text", ""))),
            source=source,
            text=str(row.get("text", "")),
            metadata={k: v for k, v in row.items() if k not in {"id", "text"}},
        )
        for row in rows
    ]
    storage.save_corpus_items(audit_id, items)
    log_decision(
        audit_id,
        "ingest",
        {"corpus_source": corpus_source, "limit": limit},
        {"count": len(items)},
        "loaded corpus via MCP adapter",
    )
    return items


def build_regions(
    items: list[CorpusItem], reduced_30d: np.ndarray, labels: np.ndarray
) -> list[Region]:
    non_noise = [int(label) for label in labels if int(label) != -1]
    total_clustered = len(non_noise)
    counts = Counter(non_noise)
    members_by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        if int(label) != -1:
            members_by_label[int(label)].append(index)

    regions: list[Region] = []
    for label in sorted(members_by_label):
        indices = members_by_label[label]
        if len(indices) < 3:
            continue
        vectors = l2_normalize(reduced_30d[indices])
        centroid = l2_normalize(vectors.mean(axis=0))
        distances = [cosine_distance(centroid, vector, clamp=True) for vector in vectors]
        exemplar_order = np.argsort(distances)[:5]
        exemplar_ids = [items[indices[int(i)]].id for i in exemplar_order]
        heterogeneity = _heterogeneity(vectors)
        label_text = _label_from_exemplars([items[indices[int(i)]].text for i in exemplar_order])
        regions.append(
            Region(
                id=f"region_{label:02d}",
                label=label_text,
                centroid=centroid.tolist(),
                density=counts[label] / total_clustered if total_clustered else 0.0,
                heterogeneity=heterogeneity,
                member_count=len(indices),
                member_corpus_ids=[items[index].id for index in indices],
                exemplar_corpus_ids=exemplar_ids,
            )
        )
    return regions


def _heterogeneity(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 0.0
    distances = []
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            distances.append(cosine_distance(vectors[left], vectors[right], clamp=True))
    return float(np.mean(distances)) if distances else 0.0


def _label_from_exemplars(exemplars: list[str]) -> str:
    words = " ".join(exemplars).lower().split()
    candidates = [word.strip(".,!?;:#") for word in words if len(word.strip(".,!?;:#")) > 4]
    common = [word for word, _ in Counter(candidates).most_common(4)]
    return " ".join(common).title() if common else "General Support"


def run_mapper_sync(*args, **kwargs) -> dict:
    return asyncio.run(run_mapper(*args, **kwargs))
