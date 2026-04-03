"""
models/db.py — SQLAlchemy ORM models

Tables:
    Document  — one row per uploaded PDF file.
    Chunk     — one row per text chunk extracted from a Document.

Scalability notes:
    - UUIDs are used as primary keys so rows can be generated on any node
      without a central sequence (important if sharding later).
    - file_hash has a UNIQUE constraint for deduplication: re-uploading the
      same PDF is a no-op.
    - The relationship Document → Chunk uses cascade="all, delete-orphan"
      so deleting a Document automatically removes all its chunks from both
      the database and (by convention) the vector store.
    - Adding indexes on doc_id and created_at is straightforward here and
      would matter at scale; omitted for demo clarity.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _now() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class Document(Base):
    """
    Represents a single uploaded PDF file.

    Status lifecycle:
        pending  → the file has been accepted but processing has not started
        processing → text is being extracted and embedded
        ready    → all chunks are embedded and searchable
        failed   → an error occurred during ingestion
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Original filename as uploaded by the user (display only — never used
    # as a filesystem path to prevent path-traversal attacks).
    filename: Mapped[str] = mapped_column(String(255), nullable=False)

    # SHA-256 hex digest of the raw file bytes.  UNIQUE constraint prevents
    # duplicate ingestion of the same file.
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    # Processing status (see docstring above).
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")

    # Metadata populated after successful ingestion.
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    # One document → many chunks (cascade delete keeps DB consistent)
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename} status={self.status}>"


class Chunk(Base):
    """
    Represents a single text chunk extracted from a Document.

    Each chunk has:
        - The raw text (stored in the DB for keyword search and citation display)
        - A reference to its position in the source document (page, index)
        - A vector_row_index that maps this chunk to a row in the numpy
          vector matrix stored on disk.  This is the join key between the
          relational world and the vector world.
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    doc_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id"), nullable=False
    )

    # The actual text content — used for BM25 keyword search and for
    # displaying citations back to the user.
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Source location for citations.
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Row index in the numpy vector matrix (data/vectors/vectors.npy).
    # Stored as an integer so we can look up the embedding for this chunk
    # in O(1) without scanning the full matrix.
    vector_row_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Precomputed character length — useful for BM25 average-doc-length calc.
    char_length: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"<Chunk id={self.id} doc_id={self.doc_id} "
            f"page={self.page_number} idx={self.chunk_index}>"
        )
