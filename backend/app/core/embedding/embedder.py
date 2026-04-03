"""
core/embedding/embedder.py — OpenAI embedding wrapper

Responsibilities:
    - Wrap the OpenAI `text-embedding-3-small` API in a thin async interface.
    - Batch input texts to stay within OpenAI's per-request token limits and
      to reduce the total number of API calls for large document ingestions.
    - Return embeddings as a numpy float32 matrix for immediate use by the
      vector store.

Why text-embedding-3-small?
    - 1536-dimensional dense embeddings; strong multilingual performance.
    - Cost-effective and fast — well-suited for high-volume ingestion.
    - Upgrade path: swap to `text-embedding-3-large` (3072 dims) via config
      for higher retrieval accuracy on complex corpora.

Batching:
    OpenAI allows up to 2048 inputs per embedding call.  We default to a
    batch size of 128 to stay well within limits and keep payloads small.

Production note:
    For very high ingestion throughput, consider parallelising batch calls
    with asyncio.gather.  Kept sequential here to avoid rate-limit errors
    during the demo.
"""

import numpy as np
from openai import AsyncOpenAI

from app.config import get_settings

settings = get_settings()

# ── OpenAI async client (module-level singleton) ──────────────────────────────
# AsyncOpenAI is used throughout so all calls are non-blocking.
# The API key is read from settings (env var), never hard-coded.
_client = AsyncOpenAI(api_key=settings.openai_api_key)

# Maximum texts per OpenAI embedding API call.
_BATCH_SIZE = 128

# Embedding dimension for text-embedding-3-small.
# Must match the model; update here if switching to text-embedding-3-large (3072).
EMBEDDING_DIM = 1536


async def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings using OpenAI's embedding model.

    Texts are sent in batches of _BATCH_SIZE to avoid oversized payloads.
    Results are concatenated into a single (N, 1536) float32 numpy array.

    Args:
        texts: List of strings to embed.  May be empty (returns empty array).

    Returns:
        numpy array of shape (len(texts), EMBEDDING_DIM) with dtype float32.

    Raises:
        openai.APIError: Propagated from the OpenAI SDK on API errors.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]

        response = await _client.embeddings.create(
            model=settings.openai_embed_model,
            input=batch,                  # OpenAI uses `input`, not `inputs`
        )
        # response.data is a list of Embedding objects, each with a .embedding list
        batch_embeddings = [obj.embedding for obj in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype=np.float32)


async def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string.

    Convenience wrapper around embed_texts() for the common single-string
    case.  Returns a 1-D array of shape (EMBEDDING_DIM,).

    Args:
        text: The query string to embed.

    Returns:
        1-D numpy float32 array of shape (1536,).
    """
    matrix = await embed_texts([text])
    return matrix[0]  # shape (1536,)
