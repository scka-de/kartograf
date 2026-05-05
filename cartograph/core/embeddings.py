from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import cast

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray

from .math import deterministic_embedding

EMBEDDING_DIM = 768
EMBEDDING_CACHE_DIR = Path("data/embeddings")


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
    from google import genai  # type: ignore[import-not-found]

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 100):
        batch = texts[start : start + 100]
        for attempt in range(2):
            try:
                response = client.models.embed_content(
                    model="text-embedding-004",
                    contents=batch,
                )
                vectors.extend([embedding.values for embedding in response.embeddings])
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
