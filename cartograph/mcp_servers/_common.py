from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def data_path(*parts: str) -> Path:
    path = Path("data").joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def print_tool_listing(tools: dict[str, Callable[..., Any]]) -> None:
    print(json.dumps({"tools": sorted(tools)}, indent=2))


def run_stdio_server(name: str, tools: dict[str, Callable[..., Any]]) -> None:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
    except Exception:
        print_tool_listing(tools)
        return

    server = FastMCP(name)
    for tool in tools.values():
        server.tool()(tool)
    server.run()


def fixture_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "demo" / "mocks" / name
