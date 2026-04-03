"""
core/reranker.py — LLM-based chunk re-ranking via Mistral

Purpose:
    After hybrid search (semantic + BM25 + RRF), we have up to `retrieval_top_k`
    candidate chunks.  Not all are equally relevant — RRF fusion is a heuristic.
    This re-ranker asks Mistral to explicitly score each chunk's relevance to the
    query on a 0–10 scale, then re-sorts by that score.

Why LLM re-ranking?
    - A cross-encoder (e.g. ms-marco-MiniLM) would be faster but requires an
      additional model dependency (sentence-transformers), which we avoid.
    - Mistral already has full language understanding; it can judge relevance
      better than embedding cosine similarity alone.
    - We batch all chunks into a single prompt to minimise API calls.

Trade-off (demo vs. production):
    - This adds ~1–2 seconds of latency (one extra Mistral API call).
    - In production, a dedicated cross-encoder or a fine-tuned re-ranker model
      would be faster and cheaper.

Prompt strategy:
    - All chunk texts are numbered and presented in a single user message.
    - The model returns a JSON array of {chunk_number, score} pairs.
    - We parse the JSON and re-sort; invalid/missing scores default to 0.

Security:
    - Chunk texts are injected inside numbered XML-style blocks to isolate
      them from the instruction context.
"""

import json
import re
from dataclasses import dataclass

from mistralai import Mistral

from app.config import get_settings

settings = get_settings()
_client = Mistral(api_key=settings.mistral_api_key)


@dataclass
class RankedChunk:
    """A chunk after LLM re-ranking, with its relevance score."""

    chunk_id: str
    text: str
    page_number: int
    filename: str
    relevance_score: float   # 0–10 from the LLM
    similarity_score: float  # original cosine similarity from vector store


_SYSTEM_PROMPT = """\
You are a relevance judge for a document retrieval system.

Given a user query and a numbered list of text chunks, score each chunk
on how relevant it is to the query on a scale from 0 to 10:
  0  = completely irrelevant
  5  = partially relevant (mentions the topic but doesn't answer)
  10 = directly and fully answers the query

Return ONLY a JSON array like:
[{"chunk": 1, "score": 8}, {"chunk": 2, "score": 3}, ...]

Include every chunk number. No explanation, no markdown, no extra keys.
"""


async def rerank_chunks(
    query: str,
    chunks: list[dict],  # list of {chunk_id, text, page_number, filename, similarity_score}
    top_n: int | None = None,
) -> list[RankedChunk]:
    """
    Re-rank a list of retrieved chunks by their relevance to the query.

    Args:
        query:  The (rewritten) user query.
        chunks: List of chunk dicts from the retrieval pipeline.  Each must
                have keys: chunk_id, text, page_number, filename, similarity_score.
        top_n:  If provided, return only the top N results after re-ranking.
                Defaults to settings.rerank_top_n.

    Returns:
        List of RankedChunk sorted by relevance_score descending.
    """
    if top_n is None:
        top_n = settings.rerank_top_n

    if not chunks:
        return []

    # Build the prompt body: numbered chunk list
    chunk_blocks = "\n\n".join(
        f"<chunk_{i+1}>\n{c['text']}\n</chunk_{i+1}>"
        for i, c in enumerate(chunks)
    )
    user_message = (
        f"Query: {query}\n\n"
        f"Chunks to score:\n{chunk_blocks}"
    )

    response = await _client.chat.complete_async(
        model=settings.mistral_small_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=256,
    )

    raw = response.choices[0].message.content.strip()

    # Parse scores; default to 0 if parsing fails for a chunk
    scores_by_index: dict[int, float] = {}
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            score_list = json.loads(match.group())
            for item in score_list:
                idx = int(item.get("chunk", 0))
                score = float(item.get("score", 0))
                scores_by_index[idx] = max(0.0, min(10.0, score))  # clamp to [0,10]
    except (json.JSONDecodeError, ValueError, TypeError):
        # Parsing failed — all chunks get score 0 (still ordered by similarity)
        pass

    ranked = [
        RankedChunk(
            chunk_id=c["chunk_id"],
            text=c["text"],
            page_number=c["page_number"],
            filename=c["filename"],
            relevance_score=scores_by_index.get(i + 1, 0.0),
            similarity_score=c["similarity_score"],
        )
        for i, c in enumerate(chunks)
    ]

    # Sort by LLM relevance score (descending), break ties by cosine similarity
    ranked.sort(key=lambda r: (r.relevance_score, r.similarity_score), reverse=True)

    return ranked[:top_n]
