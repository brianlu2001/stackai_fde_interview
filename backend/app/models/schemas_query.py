"""
models/schemas_query.py — Pydantic request/response schemas for the query endpoint

These schemas define the API contract for:
    POST /api/v1/query — ask a question about the ingested knowledge base
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /api/v1/query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's natural-language question",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve (before re-ranking)",
    )


class Citation(BaseModel):
    """A source citation pointing to a specific chunk in a document."""

    filename: str = Field(..., description="Name of the source PDF")
    page_number: int = Field(..., description="1-based page number in the PDF")
    excerpt: str = Field(
        ..., description="Short excerpt from the chunk used as evidence"
    )
    relevance_score: float = Field(
        ..., description="LLM relevance score (0–10)"
    )
    similarity_score: float = Field(
        ..., description="Cosine similarity to the query embedding (0–1)"
    )


class QueryResponse(BaseModel):
    """
    Response body for POST /api/v1/query.

    Fields:
        answer:               The generated answer, or a refusal message.
        citations:            Source chunks used as evidence (empty on refusal).
        intent:               Detected intent class (e.g. RAG_QUERY, GREETING).
        rewritten_query:      The search-optimised form of the original query.
        sufficient_evidence:  False when similarity threshold was not met.
        flagged_sentences:    Sentences removed by the hallucination filter.
    """

    answer: str = Field(..., description="Generated answer or refusal message")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source chunks supporting the answer",
    )
    intent: str = Field(..., description="Detected query intent")
    rewritten_query: str | None = Field(
        None, description="Search-optimised rewrite of the original query"
    )
    sufficient_evidence: bool = Field(
        True,
        description="False if top chunk similarity fell below the threshold",
    )
    flagged_sentences: list[str] = Field(
        default_factory=list,
        description="Sentences removed by the hallucination filter",
    )
