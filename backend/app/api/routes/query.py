"""
api/routes/query.py — Query endpoint

Endpoint:
    POST /api/v1/query

Full pipeline executed on each query:

    1.  Intent detection      — classify the query (Mistral small)
    2.  Refusal check         — short-circuit for GREETING, CHITCHAT, PII,
                                MEDICAL, LEGAL intents
    3.  Query rewriting       — convert to search-optimised form (Mistral small)
    4.  Embedding             — embed the rewritten query (mistral-embed)
    5.  Semantic search       — cosine similarity over numpy vector store
    6.  BM25 keyword search   — custom BM25 index
    7.  RRF fusion            — Reciprocal Rank Fusion of both ranked lists
    8.  Chunk hydration       — load full chunk text + metadata from SQLite
    9.  LLM re-ranking        — score chunks for relevance (Mistral small)
    10. Citation threshold    — if best score < threshold → "insufficient evidence"
    11. Template selection    — choose prompt shape (factual/list/explanation/comparison)
    12. Answer generation     — call Mistral large with the selected prompt
    13. Hallucination filter  — remove unsupported sentences from the answer
    14. Response assembly     — build QueryResponse with citations

Every step is logged at DEBUG level for observability.
"""

import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.embedding.embedder import embed_query
from app.core.generation.generator import generate_answer
from app.core.generation.templates import select_prompt
from app.core.postprocessing.hallucination import filter_hallucinations
from app.core.query.intent import Intent, detect_intent, INTENT_RESPONSES
from app.core.query.rewriter import rewrite_query
from app.core.reranker import rerank_chunks
from app.core.search.bm25 import BM25Index
from app.core.search.hybrid import reciprocal_rank_fusion
from app.core.search.vector_store import VectorStore
from app.database import get_db
from app.models.db import Chunk, Document
from app.models.schemas_query import Citation, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the knowledge base",
    description=(
        "Runs the full RAG pipeline: intent detection → query rewriting → "
        "hybrid search → re-ranking → answer generation → hallucination filtering."
    ),
)
async def query_knowledge_base(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """Execute the full RAG query pipeline and return an answer with citations."""

    query = request.query.strip()
    logger.debug("Received query: %r", query)

    # ── Step 1: Intent detection ─────────────────────────────────────────────
    intent = await detect_intent(query)
    logger.debug("Detected intent: %s", intent)

    # ── Step 2: Non-RAG intent short-circuit ─────────────────────────────────
    if intent != Intent.RAG_QUERY:
        return QueryResponse(
            answer=INTENT_RESPONSES[intent],
            intent=intent.value,
            sufficient_evidence=True,  # not a retrieval query, so N/A
        )

    # ── Step 3: Query rewriting ──────────────────────────────────────────────
    rewritten = await rewrite_query(query)
    logger.debug("Rewritten query: %r", rewritten)

    # ── Step 4: Embed the rewritten query ────────────────────────────────────
    query_embedding = await embed_query(rewritten)  # shape (1024,)

    # ── Step 5: Semantic search ───────────────────────────────────────────────
    vector_store = VectorStore(settings.vector_store_path)
    semantic_results = vector_store.search(
        query_embedding, top_k=settings.retrieval_top_k
    )
    semantic_ids = [r.chunk_id for r in semantic_results]
    semantic_scores = {r.chunk_id: r.score for r in semantic_results}
    logger.debug("Semantic search returned %d results", len(semantic_results))

    # ── Step 6: BM25 keyword search ───────────────────────────────────────────
    bm25_index = BM25Index(settings.vector_store_path)
    keyword_results = bm25_index.search(rewritten, top_k=settings.retrieval_top_k)
    keyword_ids = [r.chunk_id for r in keyword_results]
    logger.debug("BM25 search returned %d results", len(keyword_results))

    # ── Step 7: RRF fusion ───────────────────────────────────────────────────
    fused = reciprocal_rank_fusion(
        semantic_ids=semantic_ids,
        keyword_ids=keyword_ids,
        top_k=settings.retrieval_top_k,
    )
    fused_ids = [r.chunk_id for r in fused]
    logger.debug("Fused to %d candidates after RRF", len(fused_ids))

    if not fused_ids:
        return QueryResponse(
            answer="I could not find any relevant information in the knowledge base.",
            intent=intent.value,
            rewritten_query=rewritten,
            sufficient_evidence=False,
        )

    # ── Step 8: Hydrate chunks from the database ──────────────────────────────
    # Load text + metadata for the fused candidate chunk IDs.
    chunk_rows = (
        await db.execute(
            select(Chunk, Document.filename)
            .join(Document, Chunk.doc_id == Document.id)
            .where(Chunk.id.in_(fused_ids))
        )
    ).all()

    # Build a lookup dict keyed by chunk_id for ordering later
    chunk_lookup: dict[str, dict] = {
        row.Chunk.id: {
            "chunk_id": row.Chunk.id,
            "text": row.Chunk.text,
            "page_number": row.Chunk.page_number,
            "filename": row.filename,
            # Use cosine similarity score; default to 0 if chunk came from BM25 only
            "similarity_score": semantic_scores.get(row.Chunk.id, 0.0),
        }
        for row in chunk_rows
    }

    # Preserve RRF-fused order for re-ranking input
    ordered_chunks = [chunk_lookup[cid] for cid in fused_ids if cid in chunk_lookup]

    # ── Step 9: LLM re-ranking ────────────────────────────────────────────────
    ranked_chunks = await rerank_chunks(
        query=rewritten,
        chunks=ordered_chunks,
        top_n=settings.rerank_top_n,
    )
    logger.debug("Re-ranked to %d final chunks", len(ranked_chunks))

    # ── Step 10: Citation threshold check ────────────────────────────────────
    # If the highest-confidence chunk is below our similarity threshold,
    # we refuse to generate rather than risk hallucinating.
    if not ranked_chunks or ranked_chunks[0].similarity_score < settings.similarity_threshold:
        logger.info(
            "Insufficient evidence: top similarity=%.3f < threshold=%.3f",
            ranked_chunks[0].similarity_score if ranked_chunks else 0.0,
            settings.similarity_threshold,
        )
        return QueryResponse(
            answer=(
                "Insufficient evidence: the retrieved documents do not contain "
                "enough relevant information to answer this question confidently."
            ),
            intent=intent.value,
            rewritten_query=rewritten,
            sufficient_evidence=False,
        )

    # ── Step 11: Template selection ───────────────────────────────────────────
    chunk_dicts = [
        {
            "chunk_id": rc.chunk_id,
            "text": rc.text,
            "page_number": rc.page_number,
            "filename": rc.filename,
        }
        for rc in ranked_chunks
    ]
    system_prompt, user_message = select_prompt(query, chunk_dicts)

    # ── Step 12: Answer generation ────────────────────────────────────────────
    raw_answer = await generate_answer(system_prompt, user_message)
    logger.debug("Generated answer (%d chars)", len(raw_answer))

    # ── Step 13: Hallucination filter ─────────────────────────────────────────
    # Re-embed the top chunk texts to compare against answer sentences.
    # (We already have the vectors in the store by row_index, but fetching
    # them back by UUID keeps this layer decoupled from store internals.)
    from app.core.embedding.embedder import embed_texts as _embed

    chunk_embeddings = await _embed([rc.text for rc in ranked_chunks])
    clean_answer, flagged = await filter_hallucinations(
        answer=raw_answer,
        chunk_embeddings=chunk_embeddings,
        threshold=settings.similarity_threshold,
    )
    if flagged:
        logger.info("Hallucination filter removed %d sentences", len(flagged))

    # ── Step 14: Assemble response ────────────────────────────────────────────
    citations = [
        Citation(
            filename=rc.filename,
            page_number=rc.page_number,
            # Return first 300 chars of the chunk as an excerpt
            excerpt=rc.text[:300] + ("…" if len(rc.text) > 300 else ""),
            relevance_score=round(rc.relevance_score, 2),
            similarity_score=round(rc.similarity_score, 4),
        )
        for rc in ranked_chunks
    ]

    return QueryResponse(
        answer=clean_answer,
        citations=citations,
        intent=intent.value,
        rewritten_query=rewritten,
        sufficient_evidence=True,
        flagged_sentences=flagged,
    )
