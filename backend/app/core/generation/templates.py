"""
core/generation/templates.py — Prompt templates for answer generation

Answer shaping:
    Different query types warrant different response formats.  We detect
    the likely shape of the expected answer from the (rewritten) query and
    choose an appropriate template:

    FACTUAL    — A direct question expecting a concise answer (who, what,
                 when, where, why, how many).
                 Template: concise paragraph, cite sources inline.

    LIST       — Query implies an enumeration (list, steps, types, examples).
                 Template: numbered or bulleted list.

    EXPLANATION — Query asks for a detailed explanation or summary.
                 Template: multi-paragraph prose with reasoning.

    COMPARISON — Query compares two or more things.
                 Template: structured comparison (table or side-by-side).

Each template function takes the query and context chunks, and returns a
fully-formed prompt ready to pass to the generation model.

Context injection:
    Retrieved chunk texts are injected inside <context> tags.  Instructions
    are placed both before and after the context block so the model cannot
    be easily jailbroken by content embedded in the documents.
"""

import re


def _build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for injection.

    Each chunk is labelled with its source (filename, page) so the model
    can attribute claims to specific sources.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[Source {i}: {chunk['filename']}, page {chunk['page_number']}]"
        parts.append(f"{label}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _base_instructions(query: str, context: str) -> tuple[str, str]:
    """Return (system_prompt, user_message) shared across all templates."""
    system = (
        "You are a precise document assistant. Answer ONLY using the provided "
        "context. If the context does not contain enough information to answer "
        "the question, say so explicitly — do not speculate or add knowledge "
        "from outside the provided sources.\n\n"
        "Always cite the source label (e.g. [Source 1]) when you use information "
        "from that chunk."
    )
    user = (
        f"<context>\n{context}\n</context>\n\n"
        f"Question: {query}\n\n"
        "Answer strictly from the context above. Cite sources."
    )
    return system, user


def factual_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Prompt for direct factual questions.
    Expects a concise 1–3 sentence answer with inline citations.
    """
    context = _build_context_block(chunks)
    system, user = _base_instructions(query, context)
    system += (
        "\n\nResponse format: A concise answer of 1–3 sentences. "
        "Cite sources inline using [Source N] notation."
    )
    return system, user


def list_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Prompt for enumeration queries (list of steps, types, examples, etc.).
    Expects a numbered or bulleted list.
    """
    context = _build_context_block(chunks)
    system, user = _base_instructions(query, context)
    system += (
        "\n\nResponse format: A numbered or bulleted list. "
        "Each item should be concise. "
        "Cite the relevant source after each item using [Source N]."
    )
    return system, user


def explanation_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Prompt for explanation or summary queries.
    Expects a multi-paragraph prose response.
    """
    context = _build_context_block(chunks)
    system, user = _base_instructions(query, context)
    system += (
        "\n\nResponse format: A clear explanation in 2–4 paragraphs. "
        "Use plain language. Cite sources inline using [Source N]."
    )
    return system, user


def comparison_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Prompt for comparison queries (two or more items being contrasted).
    Expects a structured side-by-side comparison.
    """
    context = _build_context_block(chunks)
    system, user = _base_instructions(query, context)
    system += (
        "\n\nResponse format: A structured comparison. "
        "Use a markdown table if comparing specific attributes, "
        "or a 'vs.' paragraph style for broader contrasts. "
        "Cite sources using [Source N]."
    )
    return system, user


# ── Template selection ────────────────────────────────────────────────────────

# Patterns that indicate the answer should be a list
_LIST_PATTERNS = re.compile(
    r"\b(list|enumerate|steps|types|examples|kinds|ways|methods|"
    r"what are|give me|show me all|how many)\b",
    re.IGNORECASE,
)

# Patterns that indicate a comparison
_COMPARISON_PATTERNS = re.compile(
    r"\b(compare|contrast|difference|vs\.?|versus|similar|better|worse|"
    r"pros and cons|advantages|disadvantages)\b",
    re.IGNORECASE,
)

# Patterns that indicate a request for explanation or summary
_EXPLANATION_PATTERNS = re.compile(
    r"\b(explain|describe|summarise|summarize|overview|what is|"
    r"how does|tell me about|elaborate)\b",
    re.IGNORECASE,
)


def select_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Choose the most appropriate prompt template based on the query text.

    Matching order (first match wins):
        1. Comparison patterns → comparison_prompt
        2. List patterns       → list_prompt
        3. Explanation patterns → explanation_prompt
        4. Default             → factual_prompt

    Args:
        query:  The original (or rewritten) user query.
        chunks: Retrieved and re-ranked context chunks.

    Returns:
        (system_prompt, user_message) tuple ready for the generation API call.
    """
    if _COMPARISON_PATTERNS.search(query):
        return comparison_prompt(query, chunks)
    if _LIST_PATTERNS.search(query):
        return list_prompt(query, chunks)
    if _EXPLANATION_PATTERNS.search(query):
        return explanation_prompt(query, chunks)
    return factual_prompt(query, chunks)
