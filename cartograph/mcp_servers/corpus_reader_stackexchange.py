"""MCP tool: fetch_questions for StackExchange sites with cache and fixture fallback."""

from __future__ import annotations

from ._common import data_path, fixture_path, load_json, run_stdio_server, save_json

STACKEXCHANGE_SITE_MAP = {
    "stackoverflow": "stackoverflow",
    "money": "money",
    "travel": "travel",
    "datascience": "datascience",
}
_LAST_MODE = "unknown"


def fetch_questions(site: str, tag: str | None = None, limit: int = 500) -> list[dict]:
    global _LAST_MODE
    resolved = STACKEXCHANGE_SITE_MAP.get(site, site)
    cache = data_path("corpora", f"se_{site}_{tag or 'none'}.json")
    cached = load_json(cache, None)
    if cached is not None:
        _LAST_MODE = "real"
        return cached[:limit]
    try:
        from stackapi import StackAPI  # type: ignore[import-not-found]

        api = StackAPI(resolved)
        kwargs = {"sort": "votes", "pagesize": min(limit, 100)}
        if tag:
            kwargs["tagged"] = tag
        response = api.fetch("questions", **kwargs)
        items = [
            {
                "id": str(item.get("question_id")),
                "text": item.get("title", ""),
                "tags": item.get("tags", []),
                "score": item.get("score", 0),
            }
            for item in response.get("items", [])
        ][:limit]
        save_json(cache, items)
        _LAST_MODE = "real"
        return items
    except Exception:
        _LAST_MODE = "mocked"
        return load_json(fixture_path("corpus_reader_stackexchange.json"), [])[:limit]


def get_mode() -> str:
    return _LAST_MODE


if __name__ == "__main__":
    run_stdio_server(
        "corpus_reader_stackexchange",
        {"fetch_questions": fetch_questions, "get_mode": get_mode},
    )
