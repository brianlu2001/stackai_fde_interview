"""
core/generation/generator.py — Final answer generation via Mistral large

Pipeline position:
    This module sits at the end of the query pipeline, after:
      retrieval → re-ranking → citation threshold check → template selection

    It takes the selected template (system + user prompt already constructed)
    and calls mistral-large to produce the final answer.

Model choice:
    mistral-large-latest is used for generation (highest quality).
    Classification/rewriting/reranking use mistral-small (cheaper, faster).

Citation threshold:
    Before calling the LLM, the caller (query route) checks whether the top
    chunk's similarity score meets `settings.similarity_threshold`.  If not,
    "insufficient evidence" is returned without ever calling this function.
    This prevents the model from hallucinating when the context is weak.
"""

from mistralai import Mistral

from app.config import get_settings

settings = get_settings()
_client = Mistral(api_key=settings.mistral_api_key)


async def generate_answer(system_prompt: str, user_message: str) -> str:
    """
    Call Mistral large to generate the final answer.

    Args:
        system_prompt: Template-specific system instructions (from templates.py).
        user_message:  Context + question (from templates.py).

    Returns:
        The generated answer string.

    Raises:
        MistralAPIException: Propagated from the SDK on API errors.
    """
    response = await _client.chat.complete_async(
        model=settings.mistral_large_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,    # low temperature for factual accuracy
        max_tokens=1024,    # enough for detailed explanations
    )
    return response.choices[0].message.content.strip()
