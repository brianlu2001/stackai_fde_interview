"""
core/query/intent.py — Query intent classification via Mistral

Purpose:
    Before running the expensive retrieval pipeline, we classify the user's
    query into one of several intents.  This lets us:
      - Skip retrieval entirely for greetings and chitchat.
      - Refuse PII, medical, and legal queries with appropriate messaging.
      - Route RAG queries into the full pipeline.

Intent taxonomy:
    RAG_QUERY       — A genuine question about the ingested documents.
                      Triggers the full retrieval pipeline.
    GREETING        — Hello, hi, thanks, etc.
                      Answered directly without retrieval.
    CHITCHAT        — General conversation unrelated to documents.
                      Answered directly without retrieval.
    REFUSAL_PII     — Query contains personal identifiable information
                      (emails, SSNs, phone numbers, names + addresses, etc.).
                      Refused with a privacy notice.
    REFUSAL_MEDICAL — Query asks for medical advice or diagnosis.
                      Refused with a medical disclaimer.
    REFUSAL_LEGAL   — Query asks for legal advice.
                      Refused with a legal disclaimer.

Implementation:
    We ask Mistral small (fast and cheap) to return a single JSON object
    with an "intent" field.  Structured output keeps parsing reliable.
    A fallback to RAG_QUERY is used if the response is malformed, so a
    classification failure never silently drops a legitimate question.

Security note:
    The user's query is injected into the prompt inside a clearly delimited
    <query> block.  Instructions are placed before and after the block to
    make prompt injection harder (a user cannot easily override the
    classification directive by embedding instructions in their query).
"""

import json
import re
from enum import Enum

from mistralai import Mistral

from app.config import get_settings

settings = get_settings()
_client = Mistral(api_key=settings.mistral_api_key)


class Intent(str, Enum):
    RAG_QUERY = "RAG_QUERY"
    GREETING = "GREETING"
    CHITCHAT = "CHITCHAT"
    REFUSAL_PII = "REFUSAL_PII"
    REFUSAL_MEDICAL = "REFUSAL_MEDICAL"
    REFUSAL_LEGAL = "REFUSAL_LEGAL"


# Pre-canned responses for non-RAG intents
INTENT_RESPONSES: dict[Intent, str] = {
    Intent.GREETING: (
        "Hello! I'm your document assistant. Upload some PDFs and ask me "
        "anything about their content."
    ),
    Intent.CHITCHAT: (
        "I'm specialised in answering questions about your uploaded documents. "
        "Feel free to ask anything about the content of those files!"
    ),
    Intent.REFUSAL_PII: (
        "I'm unable to process queries that contain personal identifiable "
        "information (names, emails, phone numbers, SSNs, etc.). "
        "Please rephrase your question without personal data."
    ),
    Intent.REFUSAL_MEDICAL: (
        "I can't provide medical advice or diagnosis. Please consult a "
        "qualified healthcare professional for medical questions."
    ),
    Intent.REFUSAL_LEGAL: (
        "I can't provide legal advice. Please consult a qualified legal "
        "professional for questions of a legal nature."
    ),
}

_SYSTEM_PROMPT = """\
You are an intent classifier for a document Q&A assistant.
Classify the user query into exactly one of these intents:

  RAG_QUERY       - a genuine question about documents or their content
  GREETING        - hello, hi, thanks, bye, or similar social openers
  CHITCHAT        - general conversation unrelated to any documents
  REFUSAL_PII     - contains personal identifiable information (emails,
                    phone numbers, SSNs, full names with addresses, etc.)
  REFUSAL_MEDICAL - asks for medical advice, diagnosis, or treatment
  REFUSAL_LEGAL   - asks for legal advice or interpretation of law

Respond with ONLY a JSON object like: {"intent": "RAG_QUERY"}
No explanation, no markdown, no extra keys.
"""


async def detect_intent(query: str) -> Intent:
    """
    Classify the user's query into an Intent enum value.

    The query is sent to Mistral small with a strict classification prompt.
    If the response cannot be parsed, we fall back to RAG_QUERY so that a
    bad classification never silently suppresses a legitimate question.

    Args:
        query: Raw user query string.

    Returns:
        The detected Intent enum value.
    """
    # Delimit the user query to resist prompt injection
    user_message = f"Classify this query:\n<query>{query}</query>"

    response = await _client.chat.complete_async(
        model=settings.mistral_small_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,   # deterministic classification
        max_tokens=32,     # we only need a tiny JSON object
    )

    raw = response.choices[0].message.content.strip()

    try:
        # Extract the first {...} JSON block from the response
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response")
        data = json.loads(match.group())
        return Intent(data["intent"])
    except (json.JSONDecodeError, KeyError, ValueError):
        # Malformed response — default to RAG_QUERY (safe fallback)
        return Intent.RAG_QUERY
