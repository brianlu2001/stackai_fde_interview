"""
core/ingestion/chunker.py — Sentence-aware recursive text chunker

Chunking strategy:
    Good retrieval requires chunks that are:
      1. Small enough  — so the embedding captures a focused concept.
      2. Large enough  — so the chunk contains a complete thought.
      3. Overlapping   — so a fact that spans a boundary appears whole
                         in at least one chunk.

    We use a layered splitting approach (recursive descent):
        1. Try to split on paragraph boundaries ("\\n\\n").
        2. If a paragraph is still too large, split on sentence endings.
        3. If a sentence is still too large, fall back to a hard character
           window with overlap.

    This preserves as much semantic coherence as possible at each level
    before falling back to cruder splits.

Overlap:
    Each chunk carries `overlap` characters of text from the previous chunk
    as a prefix.  This ensures that a sentence beginning near the end of
    chunk N also appears in full at the start of chunk N+1.

Output:
    Each Chunk carries its source page number and a sequential index,
    which are stored in the database and returned as citation metadata.
"""

from dataclasses import dataclass

from app.core.ingestion.pdf_extractor import PageText


@dataclass
class TextChunk:
    """
    A single chunk of text ready for embedding.

    Attributes:
        text:        The chunk text (may include overlap prefix).
        page_number: Source page in the original PDF (1-based).
        chunk_index: Sequential index of this chunk across the whole document.
    """

    text: str
    page_number: int
    chunk_index: int


# ── Internal helpers ─────────────────────────────────────────────────────────

def _split_on_separator(text: str, separator: str) -> list[str]:
    """Split text by separator and discard empty strings."""
    return [part.strip() for part in text.split(separator) if part.strip()]


def _merge_splits(splits: list[str], max_size: int) -> list[str]:
    """
    Greedily merge small splits into chunks of at most max_size characters.

    This avoids producing hundreds of tiny two-word chunks when a paragraph
    splits into many short sentences.
    """
    merged: list[str] = []
    current = ""

    for piece in splits:
        # Would adding this piece exceed the limit?
        candidate = (current + " " + piece).strip() if current else piece
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            # If the piece itself exceeds max_size, keep it as-is and let the
            # next split level handle it.
            current = piece

    if current:
        merged.append(current)

    return merged


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """
    Last-resort character-window split for text that cannot be broken on
    natural boundaries within the size limit.

    Args:
        text:    The text to split.
        size:    Maximum characters per chunk.
        overlap: Characters of prefix carried from the previous chunk.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        # Advance by (size - overlap) so the next chunk shares `overlap`
        # characters with the current one.
        start += max(size - overlap, 1)  # guard against infinite loop
    return chunks


def _recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Recursively split text using paragraph → sentence → character fallback.

    Returns a list of strings each no longer than chunk_size characters,
    except in pathological cases where a single word exceeds chunk_size.
    """
    if len(text) <= chunk_size:
        return [text]

    # Level 1: try paragraph boundaries
    paragraphs = _split_on_separator(text, "\n\n")
    if len(paragraphs) > 1:
        merged = _merge_splits(paragraphs, chunk_size)
        # Recurse on any merged block that is still too large
        result: list[str] = []
        for block in merged:
            result.extend(_recursive_split(block, chunk_size, overlap))
        return result

    # Level 2: try sentence endings (period/question/exclamation + space)
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        merged = _merge_splits(sentences, chunk_size)
        result = []
        for block in merged:
            result.extend(_recursive_split(block, chunk_size, overlap))
        return result

    # Level 3: hard character window — no natural boundary found
    return _hard_split(text, chunk_size, overlap)


# ── Public API ───────────────────────────────────────────────────────────────

def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[TextChunk]:
    """
    Convert a list of PageText objects into a flat list of TextChunks.

    Processing order:
        1. Each page's text is cleaned (collapsed whitespace, stripped).
        2. The text is recursively split into chunks <= chunk_size chars.
        3. Overlap is injected: each chunk carries a `overlap`-char suffix
           of the previous chunk as a prefix.
        4. Chunks are assigned a global chunk_index across the document.

    Args:
        pages:      Output of pdf_extractor.extract_pages().
        chunk_size: Target maximum characters per chunk.
        overlap:    Characters of the previous chunk to prepend as context.

    Returns:
        Ordered list of TextChunk objects ready for embedding.
    """
    all_chunks: list[TextChunk] = []
    chunk_index = 0
    prev_tail = ""  # last `overlap` characters of the previous chunk

    for page in pages:
        # Normalise whitespace: collapse runs of spaces/tabs but preserve
        # paragraph breaks (double newlines) which the splitter relies on.
        import re
        text = re.sub(r"[ \t]+", " ", page.text).strip()
        if not text:
            continue  # skip blank pages

        raw_splits = _recursive_split(text, chunk_size, overlap)

        for split_text in raw_splits:
            # Prepend the tail of the previous chunk to maintain context
            # across chunk boundaries.
            if prev_tail:
                chunk_text = (prev_tail + " " + split_text).strip()
            else:
                chunk_text = split_text

            all_chunks.append(
                TextChunk(
                    text=chunk_text,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

            # Carry the last `overlap` characters into the next chunk.
            prev_tail = split_text[-overlap:] if len(split_text) > overlap else split_text

    return all_chunks
