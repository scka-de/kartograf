"""MCP tools: read_evalset and write_evalset for ADK `.evalset.json` files."""

from __future__ import annotations

import json
from pathlib import Path

from ._common import run_stdio_server


def read_evalset(path: str) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    cases = payload if isinstance(payload, list) else payload.get("eval_cases", [])
    return [_normalize_case(case, path) for case in cases]


def write_evalset(path: str, cases: list[dict], merge: bool = True) -> dict:
    target = Path(path)
    if target.exists() and merge:
        payload = json.loads(target.read_text())
    else:
        payload = {"eval_set_id": target.stem, "name": target.stem, "eval_cases": []}
    existing = payload.get("eval_cases", []) if merge else []
    raw_cases = [case.get("raw", case) for case in cases]
    payload["eval_cases"] = existing + raw_cases
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return {"path": str(target), "count": len(payload["eval_cases"])}


def _normalize_case(case: dict, source_path: str) -> dict:
    if "data" in case:
        invocations = case.get("data", [])
        content = "\n".join(str(item.get("query", "")) for item in invocations if item.get("query"))
        return {
            "id": case.get("name") or f"case_{abs(hash(content)) % 10_000_000}",
            "content": content or json.dumps(case, sort_keys=True),
            "raw": case,
            "source_path": source_path,
        }
    conversation = case.get("conversation", [])
    content_parts = []
    for turn in conversation:
        if turn.get("role") == "user" and turn.get("content"):
            content_parts.append(str(turn.get("content", "")))
            continue
        user_content = turn.get("user_content") or turn.get("userContent") or {}
        for part in user_content.get("parts", []):
            if part.get("text"):
                content_parts.append(str(part["text"]))
    content = "\n".join(content_parts)
    if not content:
        content = case.get("input") or case.get("query") or json.dumps(case, sort_keys=True)
    case_id = case.get("eval_id") or case.get("id")
    if case_id is None:
        case_id = f"case_{abs(hash(json.dumps(case, sort_keys=True))) % 10_000_000}"
    return {
        "id": case_id,
        "content": content,
        "raw": case,
        "source_path": source_path,
    }


if __name__ == "__main__":
    run_stdio_server("eval_io_adk", {"read_evalset": read_evalset, "write_evalset": write_evalset})
