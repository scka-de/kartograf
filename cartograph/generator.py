"""LLM-grounded eval case synthesis.

For each gap region, takes the k nearest surfaces + agent system prompt + existing eval
queries, and asks Gemini for one BDD-style eval case grounded in the surfaces. Falls back
to a deterministic interpolation stub if the LLM is unavailable.

Output schema:
{
  "scenario":       short name, 5-10 words
  "given":          one-line context (user/state)
  "when":           user query (this is the actual eval input)
  "then":           expected agent behavior (tool calls + response shape)
  "grounded_in":    list of surface ids the proposal is anchored on
  "validation":     "accepted" | "rejected_redundant" | "rejected_off_spec" | "fallback_stub"
}
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
from typing import Any

from .surfaces import Surface

GENERATION_MODEL = "gemini-2.5-flash"
DEFAULT_CONCURRENCY = 4

logger = logging.getLogger(__name__)
_gemini_semaphore = asyncio.Semaphore(DEFAULT_CONCURRENCY)


async def propose_eval_case(
    surfaces: list[Surface],
    nbr_indices: list[int],
    system_prompt_sentences: list[str],
    existing_eval_queries: list[str],
) -> dict:
    nbr_surfaces = [surfaces[i] for i in nbr_indices if i < len(surfaces)]
    grounded_in = [s.id for s in nbr_surfaces]
    if not nbr_surfaces:
        return _stub_proposal(grounded_in, [], "fallback_stub")

    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        return _stub_proposal(grounded_in, [s.text for s in nbr_surfaces], "fallback_stub")

    try:
        async with _gemini_semaphore:
            raw = await _generate_with_gemini(
                nbr_surfaces, system_prompt_sentences, existing_eval_queries
            )
        parsed = _parse_bdd_response(raw)
        if not parsed:
            logger.warning(
                "Gemini response could not be parsed as BDD JSON (region grounded_in=%s)",
                grounded_in[:1],
            )
            return _stub_proposal(grounded_in, [s.text for s in nbr_surfaces], "fallback_stub")
        validation = _validate_proposal(parsed["when"], existing_eval_queries)
        return {
            **parsed,
            "grounded_in": grounded_in,
            "validation": validation,
        }
    except Exception as exc:
        logger.warning(
            "propose_eval_case fallback (%s): %s",
            type(exc).__name__,
            str(exc)[:200],
        )
        return _stub_proposal(grounded_in, [s.text for s in nbr_surfaces], "fallback_stub")


async def _generate_with_gemini(
    nbr_surfaces: list[Surface],
    system_prompt_sentences: list[str],
    existing_eval_queries: list[str],
) -> str:
    genai: Any = importlib.import_module("google.genai")
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    prompt = _build_prompt(nbr_surfaces, system_prompt_sentences, existing_eval_queries)

    def call() -> str:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )
        return response.text or ""

    return await asyncio.to_thread(call)


def _build_prompt(
    nbr_surfaces: list[Surface],
    system_prompt_sentences: list[str],
    existing_eval_queries: list[str],
) -> str:
    system_block = "\n".join(f"- {s}" for s in system_prompt_sentences[:30]) or "(none)"
    surfaces_block = "\n".join(
        f"- [{s.kind}] {s.text}" for s in nbr_surfaces
    )
    evals_block = "\n".join(f'- "{q}"' for q in existing_eval_queries[:20]) or "(none)"

    return f"""You are a senior QA engineer reviewing the test suite of an AI agent. \
A coverage analysis found a semantic region the existing tests do not exercise. Your job \
is to propose ONE eval case that probes that region.

## Agent's declared behavior (system prompt excerpts)
{system_block}

## Surfaces near the uncovered region (what the agent claims to do here)
{surfaces_block}

## Existing eval queries (do NOT duplicate these)
{evals_block}

## Task
Generate exactly one eval case in BDD format that exercises the conceptual intersection \
of the surfaces above. The "when" must be a single realistic user message in plain \
English, NOT a meta-description. The "then" must list concrete expected agent behavior \
(tool calls by name, response shape).

Output as a single JSON object, no markdown fences, no preamble:

{{
  "scenario": "short scenario name, 5-10 words",
  "given": "one-line context about user state, what's in cart, history",
  "when": "the actual user query, one or two sentences in plain English",
  "then": "expected agent behavior: tool calls by name + response shape"
}}
"""


def _parse_bdd_response(raw: str) -> dict | None:
    if not raw:
        return None
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    required = ("scenario", "given", "when", "then")
    if not all(isinstance(payload.get(k), str) and payload.get(k, "").strip() for k in required):
        return None
    return {key: payload[key].strip() for key in required}


def _validate_proposal(query: str, existing_queries: list[str]) -> str:
    q_low = query.lower().strip()
    for existing in existing_queries:
        if not existing.strip():
            continue
        if _token_overlap(q_low, existing.lower().strip()) >= 0.85:
            return "rejected_redundant"
    if len(query.split()) < 3:
        return "rejected_off_spec"
    return "accepted"


def _token_overlap(a: str, b: str) -> float:
    set_a = set(re.findall(r"\w+", a))
    set_b = set(re.findall(r"\w+", b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a), len(set_b))


def _stub_proposal(grounded_in: list[str], texts: list[str], validation: str) -> dict:
    snippet = " / ".join(t.split(".")[0].strip()[:60] for t in texts[:3] if t)
    return {
        "scenario": "Conceptual intersection probe",
        "given": "the agent has surface coverage but no eval near this region",
        "when": f"Probe: {snippet}" if snippet else "(no surface texts available)",
        "then": "the agent should respond consistently with the surfaces declared",
        "grounded_in": grounded_in,
        "validation": validation,
    }
