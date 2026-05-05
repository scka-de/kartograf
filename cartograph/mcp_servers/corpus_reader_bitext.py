"""MCP tools: list_intents, fetch_examples, get_metadata for the Bitext corpus."""

from __future__ import annotations

from ._common import data_path, fixture_path, load_json, run_stdio_server, save_json

CACHE = data_path("corpora", "bitext.json")
_LAST_MODE = "unknown"


def list_intents() -> list[str]:
    return sorted({item["intent"] for item in _load_examples()})


def fetch_examples(intent: str | None = None, limit: int = 1000, offset: int = 0) -> list[dict]:
    examples = _load_examples()
    if intent is not None:
        examples = [item for item in examples if item.get("intent") == intent]
    return examples[offset : offset + limit]


def get_metadata() -> dict:
    examples = _load_examples()
    return {
        "source": "bitext",
        "count": len(examples),
        "intents": list_intents(),
        "mode": get_mode(),
    }


def get_mode() -> str:
    if CACHE.exists():
        return "real"
    return _LAST_MODE


def _load_examples() -> list[dict]:
    global _LAST_MODE
    cached = load_json(CACHE, None)
    if cached is not None:
        _LAST_MODE = "real"
        return cached
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]

        dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        split = dataset["train"]
        examples = [
            {
                "id": f"bitext_{idx}",
                "text": row.get("instruction") or row.get("response") or str(row),
                "intent": row.get("intent", "unknown"),
                "category": row.get("category", "support"),
            }
            for idx, row in enumerate(split)
        ]
        save_json(CACHE, examples)
        _LAST_MODE = "real"
        return examples
    except Exception:
        _LAST_MODE = "mocked"
        return load_json(fixture_path("corpus_reader_bitext.json"), [])


if __name__ == "__main__":
    run_stdio_server(
        "corpus_reader_bitext",
        {
            "list_intents": list_intents,
            "fetch_examples": fetch_examples,
            "get_metadata": get_metadata,
            "get_mode": get_mode,
        },
    )
