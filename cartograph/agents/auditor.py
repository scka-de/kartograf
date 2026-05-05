from __future__ import annotations

import asyncio

import numpy as np

from cartograph.core import storage
from cartograph.core.coverage import compute_coverage, coverage_warnings
from cartograph.core.embeddings import embed_texts
from cartograph.core.math import cosine_distance, cosine_similarity, stable_cache_key
from cartograph.core.models import EvalCase, Gap, RedundantEval, Region
from cartograph.core.risk import compute_risk
from cartograph.mcp_servers import eval_io_adk

from .common import log_decision

AUDITOR_PROMPT = (
    "You are the Auditor sub-agent of Cartograph. Your job: given an audit_id with regions "
    "already populated, load the eval suite (via eval_io MCP server), embed each eval case, "
    "project it into 30d using the persisted reducer, assign each eval to its nearest region "
    "centroid (cosine distance), compute the coverage score, identify uncovered regions, "
    "rank gaps by risk score, flag redundant evals (cosine similarity > 0.85 in 768d "
    "space), and persist all results. Return a one-paragraph summary: coverage score, "
    "region counts (covered/uncovered), top 3 gaps by risk."
)


async def run_auditor(audit_id: str, eval_path: str) -> dict:
    regions = storage.load_regions(audit_id)
    cases = load_eval_suite(audit_id, eval_path)
    texts = [case.content for case in cases]
    full = await embed_texts(texts, stable_cache_key(f"evals_{audit_id}", texts))
    reducer = storage.load_reducer(audit_id, "30d")
    reduced_30d = np.asarray(reducer.transform(full), dtype=float)
    from cartograph.core.math import l2_normalize

    reduced_30d = l2_normalize(reduced_30d)
    for index, case in enumerate(cases):
        case.embedding = full[index].tolist()
        case.embedding_30d = reduced_30d[index].tolist()
    suspicious = assign_evals_to_regions(cases, regions)
    regions_by_id = {region.id: region for region in regions}
    for region in regions:
        region.eval_case_ids = []
    for case in cases:
        if case.region_id and case.region_id in regions_by_id:
            regions_by_id[case.region_id].eval_case_ids.append(case.id)
    for region in regions:
        region.eval_density = (
            len(region.eval_case_ids) / region.member_count if region.member_count else 0.0
        )
    score = compute_coverage(regions)
    eval_embeddings_30d = np.asarray([case.embedding_30d for case in cases], dtype=float)
    gaps = compute_gaps(regions, eval_embeddings_30d)
    redundant = flag_redundancies(cases)
    report = storage.load_coverage_report(audit_id)
    warnings = coverage_warnings(report.noise_fraction, regions)
    if suspicious:
        warnings.append(f"{suspicious} eval assignments had distance > 0.35")
    storage.save_regions(audit_id, regions)
    storage.save_eval_cases(audit_id, cases)
    storage.save_eval_embeddings(audit_id, full, reduced_30d)
    storage.save_gaps(audit_id, gaps)
    storage.save_redundant_evals(audit_id, redundant)
    storage.update_audit_summary(audit_id, score, report.noise_fraction, warnings)
    output = {
        "coverage_score": score,
        "covered_regions": sum(1 for region in regions if region.eval_case_ids),
        "uncovered_regions": sum(1 for region in regions if not region.eval_case_ids),
        "top_gaps": [gap.model_dump() for gap in gaps[:3]],
    }
    log_decision(
        audit_id,
        "invoke_auditor",
        {"eval_path": eval_path},
        output,
        "scored eval coverage against mapped regions",
    )
    return output


def load_eval_suite(audit_id: str, eval_path: str) -> list[EvalCase]:
    try:
        rows = eval_io_adk.read_evalset(eval_path)
    except FileNotFoundError:
        rows = _synthetic_eval_rows(eval_path)
    cases = [
        EvalCase(
            id=str(row.get("id")),
            source_path=eval_path,
            content=str(row.get("content", "")),
            raw=dict(row.get("raw", row)),
        )
        for row in rows
    ]
    storage.save_eval_cases(audit_id, cases)
    return cases


def assign_evals_to_regions(cases: list[EvalCase], regions: list[Region]) -> int:
    suspicious = 0
    for case in cases:
        if case.embedding_30d is None:
            raise ValueError("eval case missing 30d embedding")
        distances = [
            (
                region.id,
                cosine_distance(
                    np.asarray(case.embedding_30d), np.asarray(region.centroid), clamp=True
                ),
            )
            for region in regions
        ]
        if not distances:
            continue
        region_id, distance = min(distances, key=lambda item: item[1])
        case.region_id = region_id
        case.assignment_distance = distance
        if distance > 0.35:
            suspicious += 1
    return suspicious


def compute_gaps(regions: list[Region], eval_embeddings_30d: np.ndarray) -> list[Gap]:
    gaps: list[Gap] = []
    for region in regions:
        if region.eval_case_ids:
            continue
        result = compute_risk(region, eval_embeddings_30d)
        gaps.append(
            Gap(
                region_id=region.id,
                risk_score=result["risk_score"],
                components=result["components"],
                rank=0,
            )
        )
    gaps.sort(key=lambda gap: gap.risk_score, reverse=True)
    return [gap.model_copy(update={"rank": index + 1}) for index, gap in enumerate(gaps)]


def flag_redundancies(cases: list[EvalCase], threshold: float = 0.85) -> list[RedundantEval]:
    redundant: list[RedundantEval] = []
    for left in range(len(cases)):
        for right in range(left + 1, len(cases)):
            if cases[left].embedding is None or cases[right].embedding is None:
                continue
            similarity = cosine_similarity(
                np.asarray(cases[left].embedding), np.asarray(cases[right].embedding)
            )
            if similarity > threshold:
                redundant.append(
                    RedundantEval(
                        case_id=cases[right].id,
                        duplicate_of_case_id=cases[left].id,
                        similarity=similarity,
                    )
                )
    return redundant


def _synthetic_eval_rows(eval_path: str) -> list[dict]:
    samples = [
        "I want a refund for my order",
        "Where is my shipment?",
        "The product arrived damaged",
        "I need to change my delivery address",
    ]
    return [
        {
            "id": f"synthetic_eval_{index}",
            "content": sample,
            "raw": {
                "eval_id": f"synthetic_eval_{index}",
                "conversation": [{"role": "user", "content": sample}],
            },
            "source_path": eval_path,
        }
        for index, sample in enumerate(samples, start=1)
    ]


def run_auditor_sync(*args, **kwargs) -> dict:
    return asyncio.run(run_auditor(*args, **kwargs))
