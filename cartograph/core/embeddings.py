from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray

from .math import deterministic_embedding

EMBEDDING_DIM = 768


def _default_cache_dir() -> Path:
    override = os.environ.get("KARTOGRAF_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "kartograf" / "embeddings"


EMBEDDING_CACHE_DIR = _default_cache_dir()


async def embed_texts(texts: list[str], cache_key: str | None = None) -> NDArray[np.float64]:
    """
    Returns shape (len(texts), 768). Uses Gemini when configured, otherwise a
    deterministic local embedding fallback so tests and demos can run offline.
    """
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EMBEDDING_CACHE_DIR / f"{cache_key}.npy" if cache_key else None
    if cache_path and cache_path.exists():
        return cast(NDArray[np.float64], np.load(cache_path))

    load_dotenv()
    if os.getenv("GOOGLE_API_KEY"):
        try:
            matrix = await _embed_with_gemini(texts)
        except Exception:
            matrix = _embed_locally(texts)
    else:
        matrix = _embed_locally(texts)

    if cache_path:
        np.save(cache_path, matrix)
    return matrix


async def _embed_with_gemini(texts: list[str]) -> NDArray[np.float64]:
    genai: Any = importlib.import_module("google.genai")
    types_mod: Any = importlib.import_module("google.genai.types")
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    config = types_mod.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 100):
        batch = texts[start : start + 100]
        for attempt in range(2):
            try:
                response: Any = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                    config=config,
                )
                embeddings = response.embeddings or []
                vectors.extend([list(embedding.values) for embedding in embeddings])
                break
            except Exception:
                if attempt == 1:
                    raise
                await asyncio.sleep(1.0)
    return cast(NDArray[np.float64], np.asarray(vectors, dtype=float))


def _embed_locally(texts: list[str]) -> NDArray[np.float64]:
    if not texts:
        return cast(NDArray[np.float64], np.empty((0, EMBEDDING_DIM), dtype=float))
    return cast(
        NDArray[np.float64],
        np.vstack([deterministic_embedding(text, EMBEDDING_DIM) for text in texts]),
    )
