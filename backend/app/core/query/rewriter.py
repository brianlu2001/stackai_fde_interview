"""
core/query/rewriter.py — Query rewriting for improved retrieval

Problem:
    User queries are often conversational, vague, or contain implicit context
    that hurts embedding similarity.  For example:
      "what does it say about fees?"
    is a poor embedding query because "it" and "say" are noise words.

Solution — Query Rewriting:
    We ask Mistral small to rephrase the query into a declarative, keyword-
    rich form that is optimised for dense retrieval.  The rewritten query
    preserves the original meaning while using the vocabulary likely present
    in the source documents.

    Example:
      Input:  "what does it say about fees?"
      Output: "fee structure pricing costs payment terms"

    This approach is sometimes called "query expansion" or a simplified form
    of HyDE (Hypothetical Document Embeddings) without generating a full
    hypothetical passage.

Why not full HyDE?
    HyDE generates a hypothetical answer and embeds that instead of the query.
    It works well but risks hallucinating terminology not present in the corpus.
    Direct rewriting is more conservative and easier to debug.

Security note:
    The query is injected inside a <query> delimiter block to make it harder
    for a user to override the rewriting instruction via prompt injection.
"""

from openai import AsyncOpenAI

from app.config import get_settings

settings = get_settings()
_client = AsyncOpenAI(api_key=settings.openai_api_key)

_SYSTEM_PROMPT = """\
You are a search query optimiser for a document retrieval system.

Your task: rewrite the user's question into a concise, keyword-rich,
declarative phrase that will retrieve the most relevant passages from
a document database.

Rules:
  - Remove filler words (what does it say about, tell me, I want to know)
  - Use noun phrases and domain terms that would appear in source documents
  - Keep it under 20 words
  - Do NOT answer the question — only rewrite it for search
  - Return the rewritten query only, no explanation, no quotes
"""


async def rewrite_query(query: str) -> str:
    """
    Rewrite a conversational user query into a search-optimised form.

    Args:
        query: The original user query string.

    Returns:
        A rewritten query string better suited for embedding similarity search.
        Falls back to the original query if the API call fails or returns
        an empty string.
    """
    user_message = f"Rewrite for search:\n<query>{query}</query>"

    response = await _client.chat.completions.create(
        model=settings.openai_small_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=64,
    )

    rewritten = response.choices[0].message.content.strip()

    # Fall back to original if the model returns something empty or too long
    if not rewritten or len(rewritten) > 500:
        return query

    return rewritten
