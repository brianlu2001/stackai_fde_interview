# StackAI RAG — Document Intelligence Assistant

A production-minded Retrieval-Augmented Generation (RAG) pipeline built with **FastAPI** and **Mistral AI**. Upload any PDF files and ask questions in natural language — the system retrieves the most relevant passages and generates grounded, cited answers.

---

## System Design

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Frontend (Chat UI)                             │
│          Vanilla HTML/CSS/JS — served directly by FastAPI             │
│   [Demo: single file. Production: React/Next.js on CDN]              │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼─────────────────────────────────────────┐
│                         FastAPI Backend                               │
│                                                                       │
│   POST /api/v1/ingest           POST /api/v1/query                    │
│   GET  /api/v1/documents        DELETE /api/v1/documents/{id}         │
│   GET  /health                  GET  /docs  (Swagger UI)              │
└──────┬──────────────────────────────────┬────────────────────────────┘
       │                                  │
┌──────▼──────────────┐       ┌───────────▼──────────────────────────┐
│   Ingestion Layer   │       │           Query Pipeline              │
│                     │       │                                       │
│  1. Validate PDF    │       │  1. Intent Detection  (Mistral small) │
│     (magic bytes +  │       │  2. Refusal Policies  (PII/med/legal) │
│      MIME + size)   │       │  3. Query Rewriting   (Mistral small) │
│                     │       │  4. Embed query       (mistral-embed) │
│  2. Deduplicate     │       │  5. Semantic search   (numpy cosine)  │
│     (SHA-256 hash)  │       │  6. BM25 keyword      (custom)        │
│                     │       │  7. RRF fusion                        │
│  3. Extract text    │       │  8. Chunk hydration   (SQLite)        │
│     (PyMuPDF)       │       │  9. LLM re-ranking    (Mistral small) │
│                     │       │  10. Citation check   (threshold)     │
│  4. Chunk text      │       │  11. Template select  (shape detect)  │
│     (sentence-aware │       │  12. Answer generation(Mistral large) │
│      + overlap)     │       │  13. Hallucination filter             │
│                     │       │  14. Response + citations             │
│  5. Embed chunks    │       └───────────────────────────────────────┘
│     (mistral-embed) │
│                     │
│  6. Persist         │
│     ├─ SQLite DB    │
│     ├─ numpy vectors│
│     └─ BM25 index   │
└─────────────────────┘
```

### Key Design Decisions

| Component | Choice | Why | Production Upgrade |
|---|---|---|---|
| **Framework** | FastAPI | Async-native, auto OpenAPI docs, Pydantic validation | No change needed |
| **Database** | SQLite + SQLAlchemy async | Zero setup; one-line swap to PostgreSQL | Change `DATABASE_URL` in `.env` to PostgreSQL |
| **Vector storage** | numpy float32 on disk | Exact cosine search; no external service | Qdrant / Weaviate / pgvector (ANN index) |
| **Keyword search** | Custom BM25 | No search libraries allowed; BM25 is industry standard | Elasticsearch / OpenSearch |
| **Hybrid fusion** | Reciprocal Rank Fusion | Parameter-free, normalises incompatible score scales | Same algorithm works at scale |
| **Embeddings** | mistral-embed | Requirement; 1024-dim dense vectors | Swap model via config |
| **Classification** | mistral-small | Fast/cheap for intent + rewriting + reranking | Fine-tuned classifier |
| **Generation** | mistral-large | Highest factual accuracy for final answer | Same or self-hosted |
| **PDF extraction** | PyMuPDF | Fastest Python PDF lib; robust to most variants | Unstructured.io for OCR/tables |
| **Frontend** | Single HTML file | Zero build step, fully self-contained demo | React/Next.js on CDN |

### No External Search / RAG Libraries

Built from scratch as required:

- **BM25** — full probabilistic formula (k₁=1.5, b=0.75) with inverted index in `core/search/bm25.py`
- **Vector search** — cosine similarity via numpy dot product in `core/search/vector_store.py`
- **RRF fusion** — 5-line formula in `core/search/hybrid.py`

No LangChain, LlamaIndex, Haystack, FAISS, rank_bm25, sentence-transformers, or any RAG/search library.

### Security

| Concern | Mitigation |
|---|---|
| **API key** | Read from env var only via pydantic-settings; never in source code |
| **File upload** | MIME type + magic bytes (`%PDF`) validation; max size enforced; stored by UUID (not filename) |
| **Path traversal** | User-supplied filenames stored in DB only; filesystem paths always use UUID |
| **Prompt injection** | User query injected inside `<query>` delimiter blocks in all prompts |
| **PII** | Intent detector flags and refuses PII-containing queries before LLM call |
| **CORS** | Explicit origin allowlist; no wildcard `*` |
| **Logging** | API keys and file contents never written to logs |

---

## Project Structure

```
stackai-rag/
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI app, lifespan, CORS, routing
│   │   ├── config.py                    # Pydantic settings (reads .env)
│   │   ├── database.py                  # SQLAlchemy async engine + session
│   │   ├── models/
│   │   │   ├── db.py                    # ORM: Document, Chunk
│   │   │   ├── schemas_ingest.py        # Pydantic schemas for ingest endpoints
│   │   │   └── schemas_query.py         # Pydantic schemas for query endpoint
│   │   ├── api/routes/
│   │   │   ├── ingest.py                # POST /ingest, GET/DELETE /documents
│   │   │   └── query.py                 # POST /query (full 14-step pipeline)
│   │   └── core/
│   │       ├── ingestion/
│   │       │   ├── pdf_extractor.py     # PyMuPDF page-by-page extraction
│   │       │   └── chunker.py           # Sentence-aware recursive chunker
│   │       ├── embedding/
│   │       │   └── embedder.py          # Mistral embed, batched
│   │       ├── search/
│   │       │   ├── vector_store.py      # numpy cosine search + disk persistence
│   │       │   ├── bm25.py              # Custom BM25 from scratch
│   │       │   └── hybrid.py            # Reciprocal Rank Fusion
│   │       ├── query/
│   │       │   ├── intent.py            # Intent classification (Mistral)
│   │       │   └── rewriter.py          # Query rewriting (Mistral)
│   │       ├── reranker.py              # LLM-based re-ranking (Mistral)
│   │       ├── generation/
│   │       │   ├── generator.py         # Answer generation (Mistral large)
│   │       │   └── templates.py         # Prompt templates by query shape
│   │       └── postprocessing/
│   │           └── hallucination.py     # Sentence-level evidence check
│   ├── data/                            # SQLite DB + numpy vectors (gitignored)
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   └── index.html                       # Single-file chat UI
├── .env.example                         # Config template (commit this)
├── .gitignore
└── README.md
```

---

## Setup & Running

### 1. Clone and install

```bash
git clone https://github.com/brianlu2001/stackai_fde_interview.git
cd stackai_fde_interview/backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp ../.env.example .env
# Edit .env and set your MISTRAL_API_KEY
```

```env
MISTRAL_API_KEY=your_key_here
DATABASE_URL=sqlite+aiosqlite:///./data/rag.db
```

### 3. Start the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| URL | Description |
|---|---|
| `http://localhost:8000` | Chat UI |
| `http://localhost:8000/docs` | Swagger API explorer |
| `http://localhost:8000/health` | Health check |

---

## API Reference

### `POST /api/v1/ingest`

Upload one or more PDF files for ingestion.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

**Response:**
```json
{
  "results": [
    {
      "filename": "document1.pdf",
      "status": "ingested",
      "document_id": "uuid",
      "page_count": 12,
      "chunk_count": 47
    }
  ],
  "total_uploaded": 1,
  "total_ingested": 1,
  "total_skipped": 0,
  "total_failed": 0
}
```

### `POST /api/v1/query`

Query the knowledge base with a natural-language question.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main pricing tiers?", "top_k": 10}'
```

**Response:**
```json
{
  "answer": "The document describes three pricing tiers: ...",
  "citations": [
    {
      "filename": "pricing.pdf",
      "page_number": 3,
      "excerpt": "Tier 1 starts at $10/month...",
      "relevance_score": 9.0,
      "similarity_score": 0.8432
    }
  ],
  "intent": "RAG_QUERY",
  "rewritten_query": "pricing tiers cost structure",
  "sufficient_evidence": true,
  "flagged_sentences": []
}
```

### `GET /api/v1/documents`

List all ingested documents.

### `DELETE /api/v1/documents/{doc_id}`

Remove a document and all its chunks from the knowledge base.

---

## Bonus Features

| Feature | Implementation |
|---|---|
| **Citation threshold** | If top chunk cosine similarity < 0.2, returns "insufficient evidence" instead of generating |
| **Answer shaping** | Query pattern matching selects from factual / list / explanation / comparison prompt templates |
| **Hallucination filter** | Each sentence in the answer is embedded and checked against retrieved chunks; unsupported sentences are removed |
| **Query refusal** | PII, medical, and legal queries are detected and refused with appropriate messaging |
| **Deduplication** | SHA-256 hash prevents re-ingesting identical files |

---

## Libraries Used

| Library | Version | Purpose | Link |
|---|---|---|---|
| FastAPI | 0.115 | Web framework | [fastapi.tiangolo.com](https://fastapi.tiangolo.com) |
| Uvicorn | 0.34 | ASGI server | [uvicorn.org](https://www.uvicorn.org) |
| SQLAlchemy | 2.0 | Async ORM | [sqlalchemy.org](https://www.sqlalchemy.org) |
| aiosqlite | 0.21 | Async SQLite driver | [github.com/omnilib/aiosqlite](https://github.com/omnilib/aiosqlite) |
| PyMuPDF | 1.25 | PDF extraction | [pymupdf.readthedocs.io](https://pymupdf.readthedocs.io) |
| numpy | 2.2 | Vector math (cosine similarity only) | [numpy.org](https://numpy.org) |
| mistralai | 1.5 | Mistral API client | [github.com/mistralai/client-python](https://github.com/mistralai/client-python) |
| pydantic-settings | 2.8 | Settings from env vars | [docs.pydantic.dev/latest/concepts/pydantic_settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| python-multipart | 0.0.20 | Multipart file upload parsing | [github.com/Kludex/python-multipart](https://github.com/Kludex/python-multipart) |
| aiofiles | 24.1 | Async file I/O | [github.com/Tinche/aiofiles](https://github.com/Tinche/aiofiles) |
