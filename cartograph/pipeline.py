"""End-to-end audit: scan repo → embed → project → density → gaps → propose → HTML."""

from __future__ import annotations

import asyncio

import numpy as np

from cartograph.core import clustering
from cartograph.core.embeddings import embed_texts
from cartograph.core.math import stable_cache_key

from .density import compute_gap_field, nearest_surface_indices
from .evals import extract_eval_inputs
from .generator import propose_eval_case
from .report import ReportData, write_html_report
from .surfaces import extract_surfaces

SURFACE_FLOOR = 30


class SurfaceBudgetError(RuntimeError):
    """Raised when a repo declares too few surfaces to support a meaningful audit."""


async def audit_repo_async(
    target_path: str,
    output_html: str,
    surface_floor: int = SURFACE_FLOOR,
    k_neighbors: int = 5,
    alpha: float | None = None,
) -> dict:
    surfaces = extract_surfaces(target_path)
    evals = extract_eval_inputs(target_path)

    if len(surfaces) < surface_floor:
        raise SurfaceBudgetError(
            f"surface budget too thin: {len(surfaces)} < {surface_floor}. "
            "Add tool descriptions, prompt content, or public docstrings before auditing."
        )

    surface_texts = [s.text for s in surfaces]
    eval_texts = [e.text for e in evals]
    combined = surface_texts + eval_texts

    cache_key = stable_cache_key("kartograf_audit", combined)
    embeddings = await embed_texts(combined, cache_key)

    n_surfaces = len(surface_texts)
    surface_emb = embeddings[:n_surfaces]
    eval_emb = embeddings[n_surfaces:] if eval_texts else np.empty((0, embeddings.shape[1]))

    reducer = clustering.fit_visualization_reducer(embeddings)
    surface_2d = np.asarray(reducer.transform(surface_emb), dtype=float)
    eval_2d = (
        np.asarray(reducer.transform(eval_emb), dtype=float)
        if len(eval_emb)
        else np.empty((0, 2))
    )

    gap = compute_gap_field(surface_2d, eval_2d, alpha=alpha)

    region_neighbors: dict[str, list[int]] = {}
    prompt_sentences = [s.text for s in surfaces if s.kind == "prompt"]
    proposal_tasks = []
    region_ids: list[str] = []
    for region in gap.regions:
        nbrs = nearest_surface_indices(region.centroid_2d, surface_2d, surface_emb, k=k_neighbors)
        region_neighbors[region.id] = nbrs
        if nbrs:
            region_ids.append(region.id)
            proposal_tasks.append(
                propose_eval_case(surfaces, nbrs, prompt_sentences, eval_texts)
            )

    raw_proposals = await asyncio.gather(*proposal_tasks) if proposal_tasks else []
    proposed: list[dict] = [
        {**proposal, "region_id": rid}
        for rid, proposal in zip(region_ids, raw_proposals, strict=True)
    ]

    report = ReportData(
        target=target_path,
        surfaces=surfaces,
        evals=evals,
        surface_points_2d=surface_2d,
        eval_points_2d=eval_2d,
        gap=gap,
        proposed_cases=proposed,
        region_neighbors=region_neighbors,
    )
    output = write_html_report(report, output_html)
    return {
        "html": str(output),
        "coverage": gap.coverage_score,
        "surface_count": len(surfaces),
        "eval_count": len(evals),
        "region_count": len(gap.regions),
        "alpha": gap.alpha,
        "bandwidth": gap.bandwidth,
        "regions": [
            {
                "id": r.id,
                "mass": r.integrated_mass,
                "centroid": list(r.centroid_2d),
            }
            for r in gap.regions
        ],
    }


def audit_repo(
    target_path: str,
    output_html: str,
    surface_floor: int = SURFACE_FLOOR,
    k_neighbors: int = 5,
    alpha: float | None = None,
) -> dict:
    return asyncio.run(
        audit_repo_async(
            target_path,
            output_html,
            surface_floor=surface_floor,
            k_neighbors=k_neighbors,
            alpha=alpha,
        )
    )


