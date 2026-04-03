"""
core/search/vector_store.py — Custom numpy-based vector store

This is NOT a third-party vector database.  It is a plain numpy matrix
stored on disk, with cosine similarity implemented from scratch.

Design:
    - All embeddings live in a single float32 matrix: shape (N, 1024).
    - A parallel JSON list maps each row index → chunk UUID.
    - On startup the matrix and index are loaded from disk into memory.
    - Writes (add / delete) update the in-memory state and flush to disk.

Cosine similarity:
    cos(q, d) = (q · d) / (‖q‖ · ‖d‖)
    Implemented with numpy dot and linalg.norm — no search library.

Why not FAISS / Annoy / Hnswlib?
    - Assignment constraint: no external search libraries.
    - Exact cosine search is O(N·D).  For thousands of chunks this is
      fast enough (< 5 ms on a laptop for 10k × 1024 floats).

Production upgrade path:
    At millions of vectors, replace this class with a Qdrant / Weaviate
    client.  The interface (add_embeddings / search / delete_by_ids) stays
    the same — only this file changes.

Thread safety:
    FastAPI runs in an async event loop (single thread for Python code).
    All methods are synchronous (no blocking I/O after the initial load)
    so they are safe to call from async routes without a lock.  If you
    move to a multi-process deployment, protect disk writes with a
    process-level lock or move to a real vector DB.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    """A single result returned by the vector store search."""

    chunk_id: str       # UUID of the matching chunk (joins to DB)
    score: float        # cosine similarity in [-1, 1]; higher is better


class VectorStore:
    """
    In-memory numpy vector store with disk persistence.

    Attributes:
        store_dir: Directory where vectors.npy and chunk_ids.json live.
    """

    # File names within store_dir
    _VECTORS_FILE = "vectors.npy"
    _IDS_FILE = "chunk_ids.json"

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state — loaded from disk on init.
        # _matrix shape: (N, embedding_dim) or (0, 1024) when empty.
        self._matrix: np.ndarray = np.empty((0, 1024), dtype=np.float32)
        # Parallel list: _chunk_ids[i] is the chunk UUID for row i.
        self._chunk_ids: list[str] = []

        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load matrix and ID list from disk if they exist."""
        vectors_path = self.store_dir / self._VECTORS_FILE
        ids_path = self.store_dir / self._IDS_FILE

        if vectors_path.exists() and ids_path.exists():
            self._matrix = np.load(str(vectors_path))
            with open(ids_path) as f:
                self._chunk_ids = json.load(f)
        # else: stays empty — first run or clean state

    def _flush(self) -> None:
        """Persist the current in-memory state to disk."""
        np.save(str(self.store_dir / self._VECTORS_FILE), self._matrix)
        with open(self.store_dir / self._IDS_FILE, "w") as f:
            json.dump(self._chunk_ids, f)

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_embeddings(
        self, chunk_ids: list[str], embeddings: np.ndarray
    ) -> list[int]:
        """
        Append new embeddings to the store.

        Args:
            chunk_ids:  Ordered list of chunk UUIDs to register.
            embeddings: Float32 array of shape (len(chunk_ids), embedding_dim).

        Returns:
            List of row indices assigned to each chunk (for storing in the DB).

        Raises:
            ValueError: If lengths don't match or embeddings have wrong shape.
        """
        if len(chunk_ids) != len(embeddings):
            raise ValueError(
                f"chunk_ids length ({len(chunk_ids)}) != embeddings rows ({len(embeddings)})"
            )
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-D array (N, dim)")

        first_new_row = len(self._chunk_ids)

        if self._matrix.shape[0] == 0:
            self._matrix = embeddings.astype(np.float32)
        else:
            self._matrix = np.vstack([self._matrix, embeddings.astype(np.float32)])

        self._chunk_ids.extend(chunk_ids)
        self._flush()

        return list(range(first_new_row, first_new_row + len(chunk_ids)))

    def delete_by_ids(self, chunk_ids_to_remove: set[str]) -> None:
        """
        Remove chunks from the store by their UUIDs.

        Rows are removed by rebuilding the matrix without the deleted rows.
        This is O(N) but acceptable given the demo scale; a real vector DB
        supports O(log N) deletes.

        Args:
            chunk_ids_to_remove: Set of chunk UUIDs to delete.
        """
        keep_mask = [
            cid not in chunk_ids_to_remove for cid in self._chunk_ids
        ]
        if not any(keep_mask):
            self._matrix = np.empty((0, self._matrix.shape[1] if self._matrix.ndim == 2 else 1024), dtype=np.float32)
            self._chunk_ids = []
        else:
            keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
            self._matrix = self._matrix[keep_indices]
            self._chunk_ids = [self._chunk_ids[i] for i in keep_indices]

        self._flush()

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """
        Return the top_k most similar chunks using cosine similarity.

        Algorithm:
            1. Normalise the query vector to unit length.
            2. Normalise all stored vectors row-wise.
            3. Cosine similarity = dot product of normalised vectors.
            4. argsort descending → take top_k.

        All computation is numpy — no search library involved.

        Args:
            query_vector: 1-D float32 array of shape (embedding_dim,).
            top_k:        Maximum number of results to return.

        Returns:
            List of SearchResult sorted by score descending.
        """
        if self._matrix.shape[0] == 0:
            return []

        # Normalise query
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            return []
        q_unit = query_vector / q_norm

        # Normalise all stored vectors row-wise (shape: N, dim)
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        # Replace zero norms with 1 to avoid division by zero (those rows
        # will produce a score of 0, which is correct).
        norms = np.where(norms == 0, 1.0, norms)
        normalised = self._matrix / norms

        # Cosine similarity: (N, dim) @ (dim,) → (N,)
        scores = normalised @ q_unit

        # Top-k indices (unsorted), then sort descending
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            SearchResult(chunk_id=self._chunk_ids[i], score=float(scores[i]))
            for i in top_indices
        ]

    # ── Inspection ───────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        """Total number of embeddings currently stored."""
        return len(self._chunk_ids)
