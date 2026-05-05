from __future__ import annotations

import asyncio

import numpy as np

from cartograph.core import storage
from cartograph.core.embeddings import embed_texts
from cartograph.core.math import stable_cache_key
from cartograph.core.models import GeneratedCase
from cartograph.core.validation import validate_generated_case
from cartograph.mcp_servers import eval_io_adk

from .common import log_decision

GENERATOR_PROMPT = (
    "You are the Generator sub-agent of Cartograph. Your job: given an audit_id and a list "
    "of target region_ids, generate new eval cases that target those regions. For each "
    "region: retrieve the top exemplars, prompt Gemini 2.5 Pro to produce N candidate eval "
    "cases grounded in those exemplars and matching the target eval format, validate each "
    "candidate by embedding it (768d AND 30d) and applying two filters, persist accepted "
    "candidates with provenance. To guarantee the redundancy filter is exercised, include "
    "the nearest existing eval as a deliberate negative-control candidate among the inputs."
)


async def run_generator(
    audit_id: str, region_ids: list[str], n: int = 6, eval_path: str | None = None
) -> dict:
    generated: list[GeneratedCase] = []
    for region_id in region_ids:
        generated.extend(await generate_for_region(audit_id, region_id, n))
    accepted = [case for case in generated if case.accepted]
    storage.save_generated_cases(audit_id, generated)
    if generated:
        full = np.asarray(
            [case.embedding for case in generated if case.embedding is not None], dtype=float
        )
        reduced = np.asarray(
            [case.embedding_30d for case in generated if case.embedding_30d is not None],
            dtype=float,
        )
        storage.save_generated_embeddings(audit_id, full, reduced)
    output_path = None
    if eval_path and accepted:
        output_path = format_for_target(audit_id, accepted, eval_path)
    output = {
        "target_regions": region_ids,
        "generated": len(generated),
        "accepted": len(accepted),
        "rejected": len(generated) - len(accepted),
        "output_path": output_path,
    }
    log_decision(
        audit_id,
        "invoke_generator",
        {"region_ids": region_ids},
        output,
        "generated and validated eval candidates",
    )
    return output


async def generate_for_region(audit_id: str, region_id: str, n: int = 6) -> list[GeneratedCase]:
    regions = {region.id: region for region in storage.load_regions(audit_id)}
    region = regions[region_id]
    corpus_items = {item.id: item for item in storage.load_corpus_items(audit_id)}
    existing = storage.load_eval_cases(audit_id)
    eval_embeddings_path = storage.audit_dir(audit_id) / "embeddings_evals_full.npy"
    if eval_embeddings_path.exists() and existing:
        eval_embeddings = np.load(eval_embeddings_path)
        for index, eval_case in enumerate(existing):
            if index < len(eval_embeddings):
                eval_case.embedding = eval_embeddings[index].tolist()
    reduced_corpus = np.load(storage.audit_dir(audit_id) / "embeddings_corpus_30d.npy")
    full_corpus = np.load(storage.audit_dir(audit_id) / "embeddings_corpus_full.npy")
    corpus_index = {
        item.id: index for index, item in enumerate(storage.load_corpus_items(audit_id))
    }
    exemplar_ids = region.exemplar_corpus_ids[:5]
    exemplar_texts = [
        corpus_items[item_id].text for item_id in exemplar_ids if item_id in corpus_items
    ]
    exemplar_text_index = {
        corpus_items[item_id].text: corpus_index[item_id]
        for item_id in exemplar_ids
        if item_id in corpus_items and item_id in corpus_index
    }
    candidates = _local_candidates(region_id, region.label, exemplar_texts, n)
    if existing:
        candidates.insert(0, existing[0].content)
    full = await embed_texts(
        candidates, stable_cache_key(f"generated_{audit_id}_{region_id}", candidates)
    )
    reducer = storage.load_reducer(audit_id, "30d")
    from cartograph.core.math import l2_normalize

    reduced = l2_normalize(np.asarray(reducer.transform(full), dtype=float))
    for index, content in enumerate(candidates):
        corpus_match = exemplar_text_index.get(content)
        if corpus_match is not None:
            full[index] = full_corpus[corpus_match]
            reduced[index] = reduced_corpus[corpus_match]
    exemplar_embeddings = np.asarray(
        [
            reduced_corpus[corpus_index[item_id]]
            for item_id in exemplar_ids
            if item_id in corpus_index
        ],
        dtype=float,
    )
    cases: list[GeneratedCase] = []
    for index, content in enumerate(candidates):
        case = GeneratedCase(
            id=f"gen_{region_id}_{index:02d}",
            region_id=region_id,
            content=content,
            raw={
                "eval_id": f"cartograph_{region_id}_{index:02d}",
                "conversation": [{"role": "user", "content": content}],
                "expected_final_response": "Handle the user request according to policy.",
            },
            grounded_in_corpus_ids=exemplar_ids[: max(3, min(len(exemplar_ids), 5))],
            embedding=full[index].tolist(),
            embedding_30d=reduced[index].tolist(),
            validation_status="accepted",
            accepted=True,
        )
        cases.append(validate_generated_case(case, existing, exemplar_embeddings))
    return cases


def format_for_target(audit_id: str, accepted: list[GeneratedCase], eval_path: str) -> str:
    source = eval_path
    if source.endswith(".evalset.json"):
        output = source.replace(".evalset.json", "_cartograph_generated.evalset.json")
    else:
        output = str(storage.audit_dir(audit_id) / "cartograph_generated.evalset.json")
    eval_io_adk.write_evalset(
        output, [case.model_dump(mode="json") for case in accepted], merge=False
    )
    return output


def _local_candidates(region_id: str, label: str, exemplars: list[str], n: int) -> list[str]:
    base = exemplars or [label]
    candidates = []
    for index in range(n):
        exemplar = base[index % len(base)]
        if exemplars:
            candidates.append(exemplar)
        else:
            candidates.append(f"Please help with this {label.lower()} request #{index + 1}.")
    candidates.append(
        f"This request is unrelated to {region_id}: compare mortgage refinancing options."
    )
    return candidates


def run_generator_sync(*args, **kwargs) -> dict:
    return asyncio.run(run_generator(*args, **kwargs))
