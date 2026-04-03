"""
main.py — FastAPI application entry point

Responsibilities:
    - Create the FastAPI app instance with metadata for auto-generated docs.
    - Register the lifespan context manager (startup / shutdown hooks).
    - Configure CORS middleware with an explicit allowed-origins list.
    - Mount the frontend static file at the root URL.
    - Register all API routers.

Lifespan:
    On startup:
        - Creates the data/ directory if it doesn't exist.
        - Runs `init_db()` to create all database tables (idempotent).
        - Logs a startup confirmation.
    On shutdown:
        - Disposes the SQLAlchemy async engine (closes all connections).

CORS:
    Allowed origins are read from settings.allowed_origins_list (set in .env).
    This is intentionally not a wildcard ("*") — only the origins that
    actually serve the frontend are permitted.

Frontend:
    The single-file UI (frontend/index.html) is served directly by FastAPI
    from the /  path.  This keeps the demo self-contained (one `uvicorn`
    command starts everything).  In production, the frontend would be
    deployed separately to a CDN.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import ingest, query
from app.config import get_settings
from app.database import engine, init_db

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager executed once at startup and once at shutdown.

    Startup:
        1. Ensure the data directory exists (SQLite file + numpy vectors live here).
        2. Initialise the database schema (creates tables if absent).

    Shutdown:
        1. Dispose the SQLAlchemy engine to close all open DB connections cleanly.
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    # Resolve the data directory relative to this file's location so the
    # server works regardless of the working directory it's launched from.
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    vectors_dir = data_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initialising database…")
    await init_db()
    logger.info("Database ready.")
    logger.info("RAG server started. Docs at http://127.0.0.1:8000/docs")

    yield  # server is running while we wait here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down — disposing database engine…")
    await engine.dispose()


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="StackAI RAG API",
    description=(
        "A retrieval-augmented generation pipeline over uploaded PDF files. "
        "Powered by Mistral AI, custom BM25, numpy vector search, and FastAPI."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# No wildcard (*) — only the configured origins are allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingest.router)
app.include_router(query.router)

# ── Frontend ──────────────────────────────────────────────────────────────────
# Serve the single-file UI at the root path.
# The frontend/ directory sits one level above backend/.
_FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
_INDEX_HTML = _FRONTEND_DIR / "index.html"


@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serve the chat UI at the root URL."""
    if not _INDEX_HTML.exists():
        from fastapi.responses import JSONResponse
        return JSONResponse({"message": "Frontend not found. API is running at /docs"})
    return FileResponse(str(_INDEX_HTML))


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"], summary="Health check")
async def health() -> dict:
    """Returns 200 OK when the server is running."""
    return {"status": "ok", "version": "1.0.0"}
