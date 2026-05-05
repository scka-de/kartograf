"""MCP tools for generic YAML and JSONL eval file round-tripping."""

from __future__ import annotations

import json
from pathlib import Path

from ._common import run_stdio_server


def read_yaml(path: str) -> list[dict]:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("pyyaml is required for YAML eval I/O") from exc
    payload = yaml.safe_load(Path(path).read_text()) or []
    return payload if isinstance(payload, list) else payload.get("cases", [])


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_yaml(path: str, cases: list[dict]) -> dict:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("pyyaml is required for YAML eval I/O") from exc
    Path(path).write_text(yaml.safe_dump(cases, sort_keys=False))
    return {"path": path, "count": len(cases)}


def write_jsonl(path: str, cases: list[dict]) -> dict:
    Path(path).write_text("\n".join(json.dumps(case) for case in cases) + "\n")
    return {"path": path, "count": len(cases)}


if __name__ == "__main__":
    run_stdio_server(
        "eval_io_git",
        {
            "read_yaml": read_yaml,
            "read_jsonl": read_jsonl,
            "write_yaml": write_yaml,
            "write_jsonl": write_jsonl,
        },
    )
