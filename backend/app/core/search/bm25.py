"""
core/search/bm25.py — BM25 keyword search implemented from scratch

BM25 (Best Match 25) is the standard probabilistic retrieval model used by
Elasticsearch, Lucene, and most production search engines.  We implement
it here without any external library (rank_bm25, whoosh, etc.).

Algorithm:
    score(q, d) = Σ_{t in q} IDF(t) · (tf(t,d) · (k1+1)) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))

    Where:
        tf(t, d)  = term frequency of token t in document d
        IDF(t)    = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        |d|       = length of document d (in tokens)
        avgdl     = average document length across the corpus
        k1        = term saturation parameter (default 1.5)
        b         = length normalisation parameter (default 0.75)

Tokeniser:
    Lowercase + split on non-alphanumeric characters.  Simple and fast;
    a production system would use language-aware stemming/lemmatisation
    (e.g. spaCy, NLTK) but that adds dependencies and complexity beyond
    the demo scope.

Index structure:
    - inverted_index: dict[token → dict[chunk_id → tf]]
    - doc_lengths:    dict[chunk_id → token_count]
    - chunk_texts:    dict[chunk_id → raw_text]  (for result display)
    Stored in memory; serialised to JSON for persistence.

Production upgrade:
    Replace with Elasticsearch for sharding, language analysers, and
    sub-millisecond search on billions of tokens.
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BM25Result:
    """A single BM25 search result."""

    chunk_id: str
    score: float


def _tokenise(text: str) -> list[str]:
    """
    Lowercase and split on non-alphanumeric characters.

    Returns a list of tokens; empty strings are filtered out.
    """
    return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]


class BM25Index:
    """
    In-memory BM25 index with optional disk persistence.

    Usage:
        index = BM25Index()
        index.add_documents({"chunk-uuid": "The quick brown fox..."})
        results = index.search("quick fox", top_k=5)
    """

    # BM25 hyperparameters (industry standard defaults)
    K1: float = 1.5   # controls term frequency saturation
    B: float = 0.75   # controls length normalisation (0 = none, 1 = full)

    _INDEX_FILE = "bm25_index.json"

    def __init__(self, store_dir: str | Path | None = None) -> None:
        """
        Args:
            store_dir: If provided, the index is loaded from and flushed to
                       this directory.  If None, the index is memory-only.
        """
        self.store_dir = Path(store_dir) if store_dir else None
        if self.store_dir:
            self.store_dir.mkdir(parents=True, exist_ok=True)

        # token → {chunk_id → term_frequency}
        self._inverted: dict[str, dict[str, int]] = {}
        # chunk_id → token count for that chunk
        self._doc_lengths: dict[str, int] = {}
        # Total number of tokens across all documents (for avgdl calculation)
        self._total_tokens: int = 0

        if self.store_dir:
            self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load index from JSON if it exists."""
        path = self.store_dir / self._INDEX_FILE
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self._inverted = data.get("inverted", {})
            self._doc_lengths = data.get("doc_lengths", {})
            self._total_tokens = data.get("total_tokens", 0)

    def _flush(self) -> None:
        """Persist index to JSON."""
        if not self.store_dir:
            return
        path = self.store_dir / self._INDEX_FILE
        with open(path, "w") as f:
            json.dump(
                {
                    "inverted": self._inverted,
                    "doc_lengths": self._doc_lengths,
                    "total_tokens": self._total_tokens,
                },
                f,
            )

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_documents(self, documents: dict[str, str]) -> None:
        """
        Index a batch of documents.

        Args:
            documents: Mapping of chunk_id → text.  Existing chunk_ids are
                       skipped (idempotent — safe to call on re-ingestion).
        """
        for chunk_id, text in documents.items():
            if chunk_id in self._doc_lengths:
                continue  # already indexed

            tokens = _tokenise(text)
            self._doc_lengths[chunk_id] = len(tokens)
            self._total_tokens += len(tokens)

            # Build term-frequency map for this document
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            # Update inverted index
            for token, freq in tf.items():
                if token not in self._inverted:
                    self._inverted[token] = {}
                self._inverted[token][chunk_id] = freq

        self._flush()

    def remove_documents(self, chunk_ids: set[str]) -> None:
        """
        Remove documents from the index by chunk ID.

        Args:
            chunk_ids: Set of chunk UUIDs to remove.
        """
        for chunk_id in chunk_ids:
            if chunk_id not in self._doc_lengths:
                continue

            length = self._doc_lengths.pop(chunk_id)
            self._total_tokens -= length

            # Remove from inverted index
            tokens_to_clean: list[str] = []
            for token, postings in self._inverted.items():
                postings.pop(chunk_id, None)
                if not postings:
                    tokens_to_clean.append(token)

            for token in tokens_to_clean:
                del self._inverted[token]

        self._flush()

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        """
        Score all indexed documents for the given query and return top_k.

        The BM25 score is computed as described in the module docstring.
        Documents with a score of 0 (no query tokens present) are excluded.

        Args:
            query: Raw query string (will be tokenised internally).
            top_k: Maximum number of results to return.

        Returns:
            List of BM25Result sorted by score descending.
        """
        query_tokens = _tokenise(query)
        if not query_tokens or not self._doc_lengths:
            return []

        N = len(self._doc_lengths)
        avgdl = self._total_tokens / N if N > 0 else 1.0

        scores: dict[str, float] = {}

        for token in query_tokens:
            if token not in self._inverted:
                continue

            postings = self._inverted[token]
            df = len(postings)  # document frequency for this token

            # BM25 IDF (Robertson-Sparck Jones variant — always positive)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for chunk_id, tf in postings.items():
                doc_len = self._doc_lengths[chunk_id]

                # BM25 term score
                numerator = tf * (self.K1 + 1)
                denominator = tf + self.K1 * (
                    1 - self.B + self.B * doc_len / avgdl
                )
                term_score = idf * numerator / denominator

                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        if not scores:
            return []

        # Sort by score descending and return top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [BM25Result(chunk_id=cid, score=s) for cid, s in ranked[:top_k]]

    # ── Inspection ───────────────────────────────────────────────────────────

    @property
    def total_documents(self) -> int:
        """Number of documents (chunks) currently indexed."""
        return len(self._doc_lengths)
