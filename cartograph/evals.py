"""Eval input extraction from ADK eval files in a target repo.

Supports both:
  - new schema: { "eval_set_id": ..., "eval_cases": [ { "conversation": [...] } ] }
  - old schema: [ { "name": ..., "data": [ { "query": ... } ] } ]
  - simple test schema: [ { "query": ..., "reference": ... } ]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

EVAL_FILE_PATTERNS = ("*.evalset.json", "*.test.json", "evalset.json")


@dataclass
class EvalInput:
    id: str
    text: str
    source_file: str
    case_index: int
    metadata: dict = field(default_factory=dict)


def extract_eval_inputs(repo_path: str | Path) -> list[EvalInput]:
    root = Path(repo_path).resolve()
    files: list[Path] = []
    for pattern in EVAL_FILE_PATTERNS:
        files.extend(sorted(root.rglob(pattern)))
    files = [f for f in files if "/.adk/" not in str(f) and "eval_history" not in str(f)]
    inputs: list[EvalInput] = []
    seen_ids: set[str] = set()
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rel = str(path.relative_to(root))
        for entry in _iter_eval_inputs(payload, rel):
            if entry.id in seen_ids:
                continue
            seen_ids.add(entry.id)
            inputs.append(entry)
    return inputs


def _iter_eval_inputs(payload, rel_path: str):
    if isinstance(payload, dict) and "eval_cases" in payload:
        for index, case in enumerate(payload.get("eval_cases", [])):
            text = _extract_new_schema_text(case)
            if text:
                yield EvalInput(
                    id=f"{rel_path}::case_{index}",
                    text=text,
                    source_file=rel_path,
                    case_index=index,
                    metadata={"schema": "new", "eval_id": case.get("eval_id", "")},
                )
        return
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and "data" in payload[0]:
            global_index = 0
            for group in payload:
                for case in group.get("data", []):
                    text = case.get("query", "")
                    if text:
                        yield EvalInput(
                            id=f"{rel_path}::{group.get('name', 'g')}_{global_index}",
                            text=text,
                            source_file=rel_path,
                            case_index=global_index,
                            metadata={"schema": "old"},
                        )
                        global_index += 1
            return
        for index, case in enumerate(payload):
            if isinstance(case, dict):
                text = case.get("query") or case.get("text") or case.get("input", "")
                if text:
                    yield EvalInput(
                        id=f"{rel_path}::case_{index}",
                        text=text,
                        source_file=rel_path,
                        case_index=index,
                        metadata={"schema": "test"},
                    )


def _extract_new_schema_text(case: dict) -> str:
    conversation = case.get("conversation", [])
    if not conversation:
        return ""
    first = conversation[0]
    user = first.get("user_content", {})
    parts = user.get("parts", [])
    return " ".join(p.get("text", "") for p in parts if p.get("text"))
