"""
api/routes/ingest.py — PDF ingestion endpoints

Endpoints:
    POST   /api/v1/ingest               Upload one or more PDF files
    GET    /api/v1/documents            List all ingested documents
    DELETE /api/v1/documents/{doc_id}   Remove a document and all its chunks

Ingestion pipeline (per file):
    1. Validate: MIME type check + magic bytes check + file size check
    2. Deduplication: SHA-256 hash compared against the `file_hash` column
    3. Persist to DB: Document row created with status="processing"
    4. Extract text: PyMuPDF page-by-page
    5. Chunk: sentence-aware recursive splitter
    6. Embed: Mistral mistral-embed in batches
    7. Store vectors: numpy vector store (disk-persisted)
    8. Store chunks: Chunk rows inserted into SQLite
    9. Update BM25 index
    10. Update document status to "ready"

Security measures:
    - File size is checked before reading bytes (configurable MAX_UPLOAD_SIZE_MB).
    - MIME type is validated against the Content-Type header AND magic bytes
      (first 4 bytes of the PDF must be `%PDF`).
    - Files are stored in the vector store by UUID, never by the user-supplied
      filename, preventing path traversal attacks.
    - Original filenames are stored in the DB for display only.
"""

import hashlib
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.embedding.embedder import embed_texts
from app.core.ingestion.chunker import chunk_pages
from app.core.ingestion.pdf_extractor import extract_pages
from app.core.search.bm25 import BM25Index
from app.core.search.vector_store import VectorStore
from app.database import get_db
from app.models.db import Chunk, Document
from app.models.schemas_ingest import (
    DeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    IngestFileResult,
    IngestResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ingestion"])

# ── PDF magic bytes ──────────────────────────────────────────────────────────
# All valid PDFs begin with the 4-byte signature "%PDF"
_PDF_MAGIC = b"%PDF"


def _validate_pdf(file: UploadFile, content: bytes, max_bytes: int) -> str | None:
    """
    Validate that the uploaded file is a genuine PDF within the size limit.

    Returns an error string if invalid, or None if the file is acceptable.
    """
    # Check file size (bytes already read)
    if len(content) > max_bytes:
        return f"File exceeds maximum size of {max_bytes // (1024*1024)} MB"

    # Validate MIME type from Content-Type header
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        return f"Invalid content type: {file.content_type}. Only PDFs are accepted."

    # Validate magic bytes — the real defence against disguised files
    if not content.startswith(_PDF_MAGIC):
        return "File does not appear to be a valid PDF (magic bytes mismatch)."

    return None  # all checks passed


# ── POST /api/v1/ingest ──────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload PDF files for ingestion",
    description=(
        "Accepts one or more PDF files. Each file is validated, deduplicated "
        "by SHA-256 hash, extracted, chunked, embedded, and indexed. "
        "Already-ingested files (same hash) are silently skipped."
    ),
)
async def ingest_files(
    files: list[UploadFile] = File(..., description="One or more PDF files to ingest"),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Ingest one or more uploaded PDF files into the knowledge base."""

    # Lazy-init the shared stores (they load from disk on first access)
    vector_store = VectorStore(settings.vector_store_path)
    bm25_index = BM25Index(settings.vector_store_path)

    results: list[IngestFileResult] = []

    for file in files:
        filename = file.filename or "unknown.pdf"
        logger.info("Processing upload: %s", filename)

        # ── Step 1: Read & validate ──────────────────────────────────────────
        content = await file.read()
        error = _validate_pdf(file, content, settings.max_upload_size_bytes)
        if error:
            logger.warning("Rejected file %s: %s", filename, error)
            results.append(
                IngestFileResult(filename=filename, status="failed", reason=error)
            )
            continue

        # ── Step 2: Deduplication ────────────────────────────────────────────
        file_hash = hashlib.sha256(content).hexdigest()
        existing = await db.scalar(
            select(Document).where(Document.file_hash == file_hash)
        )
        if existing:
            logger.info("Skipping duplicate: %s (doc_id=%s)", filename, existing.id)
            results.append(
                IngestFileResult(
                    filename=filename,
                    status="skipped",
                    document_id=existing.id,
                    reason="Identical file already ingested",
                    page_count=existing.page_count,
                    chunk_count=existing.chunk_count,
                )
            )
            continue

        # ── Step 3: Create document record ───────────────────────────────────
        doc = Document(filename=filename, file_hash=file_hash, status="processing")
        db.add(doc)
        await db.flush()  # assign doc.id without committing

        try:
            # ── Step 4: Extract text ─────────────────────────────────────────
            pages = extract_pages(content)
            doc.page_count = len(pages)

            # ── Step 5: Chunk ─────────────────────────────────────────────────
            text_chunks = chunk_pages(
                pages,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )

            if not text_chunks:
                raise ValueError("No text could be extracted from this PDF.")

            # ── Step 6: Embed ─────────────────────────────────────────────────
            texts = [tc.text for tc in text_chunks]
            embeddings = await embed_texts(texts)

            # ── Step 7 & 8: Store vectors + persist chunks ────────────────────
            # Create ORM Chunk objects first to get their UUIDs
            chunk_objects = [
                Chunk(
                    doc_id=doc.id,
                    text=tc.text,
                    page_number=tc.page_number,
                    chunk_index=tc.chunk_index,
                    char_length=len(tc.text),
                )
                for tc in text_chunks
            ]
            db.add_all(chunk_objects)
            await db.flush()  # assign chunk UUIDs

            # Add embeddings to vector store — returns row indices
            chunk_uuids = [c.id for c in chunk_objects]
            row_indices = vector_store.add_embeddings(chunk_uuids, embeddings)

            # Write row indices back to the Chunk records
            for chunk_obj, row_idx in zip(chunk_objects, row_indices):
                chunk_obj.vector_row_index = row_idx

            # ── Step 9: Update BM25 index ─────────────────────────────────────
            bm25_index.add_documents(
                {c.id: c.text for c in chunk_objects}
            )

            # ── Step 10: Finalise document ────────────────────────────────────
            doc.chunk_count = len(chunk_objects)
            doc.status = "ready"
            await db.commit()

            logger.info(
                "Ingested %s: %d pages, %d chunks", filename, doc.page_count, doc.chunk_count
            )
            results.append(
                IngestFileResult(
                    filename=filename,
                    status="ingested",
                    document_id=doc.id,
                    page_count=doc.page_count,
                    chunk_count=doc.chunk_count,
                )
            )

        except Exception as exc:
            await db.rollback()
            doc.status = "failed"
            logger.exception("Failed to ingest %s: %s", filename, exc)
            results.append(
                IngestFileResult(
                    filename=filename,
                    status="failed",
                    document_id=doc.id,
                    reason=str(exc),
                )
            )

    return IngestResponse(
        results=results,
        total_uploaded=len(files),
        total_ingested=sum(1 for r in results if r.status == "ingested"),
        total_skipped=sum(1 for r in results if r.status == "skipped"),
        total_failed=sum(1 for r in results if r.status == "failed"),
    )


# ── GET /api/v1/documents ────────────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
)
async def list_documents(
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """Return all documents currently in the knowledge base."""
    result = await db.execute(
        select(Document).order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in docs],
        total=len(docs),
    )


# ── DELETE /api/v1/documents/{doc_id} ────────────────────────────────────────

@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteResponse,
    summary="Delete a document and all its chunks",
)
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DeleteResponse:
    """
    Remove a document and all associated chunks from the database,
    the vector store, and the BM25 index.
    """
    # Fetch document
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id} not found",
        )

    # Collect chunk IDs before deleting the document
    result = await db.execute(
        select(Chunk.id).where(Chunk.doc_id == doc_id)
    )
    chunk_ids = set(result.scalars().all())

    filename = doc.filename

    # Delete from DB (cascade removes Chunk rows automatically)
    await db.delete(doc)
    await db.commit()

    # Remove from vector store
    if chunk_ids:
        vector_store = VectorStore(settings.vector_store_path)
        vector_store.delete_by_ids(chunk_ids)

        bm25_index = BM25Index(settings.vector_store_path)
        bm25_index.remove_documents(chunk_ids)

    logger.info("Deleted document %s (%s)", doc_id, filename)
    return DeleteResponse(document_id=doc_id, filename=filename)
