from __future__ import annotations

from typing import Any

from .auditor import AUDITOR_PROMPT, run_auditor_sync
from .generator import GENERATOR_PROMPT, run_generator_sync
from .mapper import MAPPER_PROMPT, run_mapper_sync
from .root import ROOT_PROMPT, decide_generate_target, decide_terminate, format_final_report


def is_adk_available() -> bool:
    try:
        _load_llm_agent()
        return True
    except Exception:
        return False


def build_cartograph_adk_agent(model: str = "gemini-2.5-pro") -> Any:
    """Build the real ADK multi-agent hierarchy when google-adk is installed."""
    LlmAgent = _load_llm_agent()
    mapper = LlmAgent(
        name="mapper",
        model=model,
        description="Maps real-world corpus items into semantic regions.",
        instruction=MAPPER_PROMPT,
        tools=[run_mapper_tool],
    )
    auditor = LlmAgent(
        name="auditor",
        model=model,
        description="Scores eval-suite semantic coverage and ranks gaps.",
        instruction=AUDITOR_PROMPT,
        tools=[run_auditor_tool],
    )
    generator = LlmAgent(
        name="generator",
        model=model,
        description="Generates and validates corpus-grounded eval cases.",
        instruction=GENERATOR_PROMPT,
        tools=[run_generator_tool],
    )
    return LlmAgent(
        name="cartograph",
        model=model,
        description="Autonomous semantic coverage auditor for agent eval suites.",
        instruction=ROOT_PROMPT,
        tools=[
            decide_generate_target,
            decide_terminate,
            format_final_report,
        ],
        sub_agents=[mapper, auditor, generator],
    )


def run_mapper_tool(
    audit_id: str,
    corpus_source: str,
    corpus_limit: int = 1000,
) -> dict:
    """Invoke Cartograph Mapper for an existing audit."""
    return run_mapper_sync(audit_id, corpus_source, corpus_limit)


def run_auditor_tool(audit_id: str, eval_path: str) -> dict:
    """Invoke Cartograph Auditor for an existing audit."""
    return run_auditor_sync(audit_id, eval_path)


def run_generator_tool(audit_id: str, region_ids: list[str], eval_path: str | None = None) -> dict:
    """Invoke Cartograph Generator for target regions."""
    return run_generator_sync(audit_id, region_ids, eval_path=eval_path)


def _load_llm_agent() -> Any:
    try:
        from google.adk.agents import LlmAgent
    except Exception:
        from google.adk.agents.llm_agent import LlmAgent
    return LlmAgent
