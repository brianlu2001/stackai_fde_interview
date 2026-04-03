"""
config.py — Application settings

All configuration is read from environment variables (or a local .env file).
The API key and other secrets are NEVER hardcoded here.

pydantic-settings validates every field at startup and raises immediately
if a required value (like OPENAI_API_KEY) is missing, so the server
never starts in a broken state.

Production note:
    Replace the .env file with injected environment variables from your
    secret manager (AWS Secrets Manager, GCP Secret Manager, etc.).
    No code changes required.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central settings object.  All values come from environment variables;
    the .env file is loaded automatically in development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",          # loaded only when file exists (dev)
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",           # silently ignore unknown env vars
    )

    # ── OpenAI ──────────────────────────────────────────────────────────────
    # Key is kept in env / secret manager — never in source code.
    openai_api_key: str

    # Model assignments by task (cheap/fast for classification, large for generation)
    openai_embed_model: str = "text-embedding-3-small"  # 1536-dim, fast, cost-effective
    openai_small_model: str = "gpt-4o-mini"             # intent, rewrite, rerank
    openai_large_model: str = "gpt-4o"                  # final answer generation

    # ── Database ────────────────────────────────────────────────────────────
    # SQLite by default — swap to PostgreSQL with one line in .env:
    #   DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/ragdb
    database_url: str = "sqlite+aiosqlite:///./data/rag.db"

    # ── Vector store ────────────────────────────────────────────────────────
    # Directory (relative to the backend/ working directory) where numpy
    # vectors are persisted.  Each document shard is a separate .npy file.
    vector_store_path: str = "data/vectors"

    # ── File upload ─────────────────────────────────────────────────────────
    max_upload_size_mb: int = 20   # hard cap per file

    # ── Chunking ────────────────────────────────────────────────────────────
    chunk_size: int = 500          # target characters per chunk
    chunk_overlap: int = 100       # overlap to avoid cutting facts at boundaries

    # ── Retrieval ───────────────────────────────────────────────────────────
    retrieval_top_k: int = 10      # candidates from each retriever (semantic + BM25)
    rerank_top_n: int = 5          # final chunks passed to the generator

    # Minimum cosine similarity for the top chunk.
    # Below this → return "insufficient evidence" instead of hallucinating.
    similarity_threshold: float = 0.35

    # ── CORS ────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins.  No wildcard (*) in production.
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse the comma-separated ALLOWED_ORIGINS string into a list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    @lru_cache ensures the .env file is read exactly once at startup.
    Use FastAPI's Depends(get_settings) to inject settings into routes
    without creating new instances on every request.
    """
    return Settings()
