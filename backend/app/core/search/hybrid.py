"""
core/search/hybrid.py — Hybrid search via Reciprocal Rank Fusion (RRF)

Problem:
    Semantic search (cosine similarity) and keyword search (BM25) each have
    blind spots:
      - Semantic: misses exact keyword matches; score depends on embedding
        quality.
      - BM25: misses paraphrases and synonyms; biased towards term frequency.

Solution — RRF:
    Combine the two ranked lists using only their *rank positions*, not their
    raw scores.  This normalises the incompatible score scales of cosine
    similarity ([-1, 1]) and BM25 (unbounded positive).

    RRF score for chunk c:
        rrf(c) = Σ_r  1 / (k + rank_r(c))

    Where:
        k       = smoothing constant (60 is the standard value from the
                  original RRF paper: Cormack, Clarke & Buettcher, 2009)
        rank_r  = 1-based rank of chunk c in retriever r
                  (chunks absent from a list are not penalised — they simply
                  don't contribute a term to the sum)

Why RRF over linear score combination?
    - No tuning required (k=60 works well out of the box).
    - Robust to outlier scores (a single very high BM25 score doesn't
      dominate the fusion).
    - Proven effective: used as a baseline in many TREC benchmarks and
      production search systems.
"""

from dataclasses import dataclass


@dataclass
class HybridResult:
    """A single result from the fused ranked list."""

    chunk_id: str
    rrf_score: float    # higher = better match across both retrievers
    semantic_rank: int | None   # rank in the semantic list (1-based), or None
    keyword_rank: int | None    # rank in the BM25 list (1-based), or None


def reciprocal_rank_fusion(
    semantic_ids: list[str],
    keyword_ids: list[str],
    top_k: int = 10,
    k: int = 60,
) -> list[HybridResult]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.

    Args:
        semantic_ids: Ordered chunk IDs from the semantic (cosine) search,
                      best match first.
        keyword_ids:  Ordered chunk IDs from the BM25 keyword search,
                      best match first.
        top_k:        Number of results to return from the fused list.
        k:            RRF smoothing constant (default 60 per original paper).

    Returns:
        List of HybridResult sorted by rrf_score descending, capped at top_k.
    """
    rrf_scores: dict[str, float] = {}
    semantic_ranks: dict[str, int] = {}
    keyword_ranks: dict[str, int] = {}

    # Accumulate RRF score from semantic ranking
    for rank, chunk_id in enumerate(semantic_ids, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        semantic_ranks[chunk_id] = rank

    # Accumulate RRF score from keyword ranking
    for rank, chunk_id in enumerate(keyword_ids, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        keyword_ranks[chunk_id] = rank

    # Sort all seen chunk IDs by their combined RRF score (descending)
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        HybridResult(
            chunk_id=cid,
            rrf_score=score,
            semantic_rank=semantic_ranks.get(cid),
            keyword_rank=keyword_ranks.get(cid),
        )
        for cid, score in ranked[:top_k]
    ]
