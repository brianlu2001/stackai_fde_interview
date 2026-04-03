"""
database.py — Async SQLAlchemy engine, session factory, and Base

Design notes:
    - Uses SQLAlchemy 2.x async API throughout.
    - The engine is created once from DATABASE_URL in settings.
    - To switch from SQLite to PostgreSQL, set DATABASE_URL in .env —
      no other code changes are required (SQLAlchemy abstracts the driver).
    - get_db() is a FastAPI dependency that yields a session per request
      and guarantees cleanup via try/finally.

Production note (scalability):
    SQLite has a single-writer lock which is fine for a demo.  For concurrent
    write workloads, set DATABASE_URL to a PostgreSQL connection string and
    add asyncpg to requirements.txt.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

# ── Engine ──────────────────────────────────────────────────────────────────
# echo=False in production; set to True for SQL query logging during dev.
# connect_args is SQLite-specific: allows the same connection to be used
# across threads (needed for async).
_connect_args = (
    {"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {}
)

engine = create_async_engine(
    settings.database_url,
    connect_args=_connect_args,
    echo=False,
)

# ── Session factory ─────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # keep ORM objects usable after commit
)


# ── Declarative base ─────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """All ORM models inherit from this base."""
    pass


# ── FastAPI dependency ───────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield a database session for a single request, then close it.

    Usage in a route:
        @router.post("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ── Schema initialisation ────────────────────────────────────────────────────
async def init_db() -> None:
    """
    Create all tables defined by ORM models if they don't already exist.
    Called once at application startup via FastAPI's lifespan hook.
    """
    # Import models here so their table definitions are registered on Base
    # before create_all runs.  The import must happen inside the function
    # to avoid circular imports at module load time.
    from app.models import db as _models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
