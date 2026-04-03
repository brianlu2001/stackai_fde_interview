"""
models/schemas_ingest.py — Pydantic request/response schemas for ingestion endpoints

These schemas define the API contract for:
    POST /api/v1/ingest          — upload one or more PDF files
    GET  /api/v1/documents       — list all ingested documents
    DELETE /api/v1/documents/{id} — remove a document and its chunks
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Represents a single document as returned by the API."""

    id: str = Field(..., description="UUID of the document")
    filename: str = Field(..., description="Original filename as uploaded")
    status: Literal["pending", "processing", "ready", "failed"] = Field(
        ..., description="Current processing status"
    )
    page_count: int | None = Field(None, description="Number of pages extracted")
    chunk_count: int | None = Field(None, description="Number of chunks created")
    created_at: datetime = Field(..., description="UTC timestamp of upload")

    model_config = {"from_attributes": True}


class IngestResponse(BaseModel):
    """
    Response body for POST /api/v1/ingest.

    Returns one entry per uploaded file, including skipped duplicates.
    """

    results: list["IngestFileResult"] = Field(
        ..., description="Per-file ingestion outcome"
    )
    total_uploaded: int = Field(..., description="Files received in this request")
    total_ingested: int = Field(..., description="Files newly ingested (not duplicates)")
    total_skipped: int = Field(..., description="Files skipped (already in the system)")
    total_failed: int = Field(..., description="Files that failed to process")


class IngestFileResult(BaseModel):
    """Outcome for a single file in an ingest request."""

    filename: str
    status: Literal["ingested", "skipped", "failed"]
    document_id: str | None = Field(
        None, description="Assigned document UUID (null if failed)"
    )
    reason: str | None = Field(
        None, description="Reason for skip or failure (null if ingested)"
    )
    page_count: int | None = None
    chunk_count: int | None = None


class DocumentListResponse(BaseModel):
    """Response body for GET /api/v1/documents."""

    documents: list[DocumentResponse]
    total: int = Field(..., description="Total number of documents in the system")


class DeleteResponse(BaseModel):
    """Response body for DELETE /api/v1/documents/{id}."""

    document_id: str
    filename: str
    message: str = "Document and all associated chunks have been deleted."
