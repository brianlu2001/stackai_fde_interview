"""
core/postprocessing/hallucination.py — Post-hoc hallucination filter

Problem:
    Even with accurate retrieval and a grounded prompt, LLMs occasionally
    "hallucinate" — producing sentences not supported by the retrieved context.
    This is especially risky when the model blends knowledge from its training
    data with information from the context.

Approach — sentence-level evidence check:
    1. Split the generated answer into individual sentences.
    2. Embed each sentence using Mistral embed.
    3. Compute cosine similarity between each sentence embedding and all
       retrieved chunk embeddings.
    4. Any sentence whose max similarity to all chunks is below a threshold
       is flagged as "unsupported".
    5. Unsupported sentences are removed from the final answer and reported
       separately so the caller can inform the user.

Threshold:
    Uses the same `similarity_threshold` setting as the citation gate.
    A lower value (e.g. 0.25) is more permissive; higher (e.g. 0.45) is
    stricter.  For the demo we use the global threshold.

Trade-offs:
    - Embedding-based similarity is an approximation; it may flag paraphrased
      but accurate sentences (false positives) or miss subtle hallucinations
      (false negatives).
    - This is a best-effort heuristic, not a guarantee of factual accuracy.
    - A more rigorous approach would use an NLI (Natural Language Inference)
      model to check entailment, but that requires additional dependencies.

Production upgrade:
    Replace the embedding similarity check with a dedicated NLI model
    (e.g. cross-encoder/nli-deberta-v3-base via sentence-transformers)
    for higher precision hallucination detection.
"""

import re

import numpy as np

from app.core.embedding.embedder import embed_texts
from app.config import get_settings

settings = get_settings()


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences on period / question mark / exclamation mark
    followed by whitespace or end-of-string.

    Keeps the punctuation attached to the preceding sentence.
    """
    # Split after sentence-ending punctuation followed by whitespace
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    # Filter out very short fragments (e.g. abbreviations split incorrectly)
    return [s.strip() for s in raw if len(s.strip()) > 10]


async def filter_hallucinations(
    answer: str,
    chunk_embeddings: np.ndarray,
    threshold: float | None = None,
) -> tuple[str, list[str]]:
    """
    Remove sentences from the answer that lack supporting evidence.

    Args:
        answer:           The raw generated answer text.
        chunk_embeddings: Float32 array of shape (N, dim) — the embeddings
                          of the retrieved chunks (already computed during
                          retrieval; passed in to avoid re-embedding).
        threshold:        Minimum cosine similarity for a sentence to be
                          considered "supported".  Defaults to
                          settings.similarity_threshold.

    Returns:
        (clean_answer, flagged_sentences) where:
            clean_answer      — answer with unsupported sentences removed.
            flagged_sentences — list of sentences that were removed.
    """
    if threshold is None:
        threshold = settings.similarity_threshold

    sentences = _split_into_sentences(answer)
    if not sentences or chunk_embeddings.shape[0] == 0:
        return answer, []

    # Embed all sentences in a single batch call
    sentence_embeddings = await embed_texts(sentences)

    # Normalise both sentence and chunk embeddings for cosine similarity
    def _normalise(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return mat / norms

    sent_norm = _normalise(sentence_embeddings)      # (S, dim)
    chunk_norm = _normalise(chunk_embeddings)         # (N, dim)

    # Similarity matrix: (S, N) — each sentence vs. each chunk
    sim_matrix = sent_norm @ chunk_norm.T

    # For each sentence, take the max similarity across all chunks
    max_similarities = sim_matrix.max(axis=1)        # (S,)

    supported: list[str] = []
    flagged: list[str] = []

    for sentence, max_sim in zip(sentences, max_similarities):
        if float(max_sim) >= threshold:
            supported.append(sentence)
        else:
            flagged.append(sentence)

    clean_answer = " ".join(supported) if supported else answer
    return clean_answer, flagged
