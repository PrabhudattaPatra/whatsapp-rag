# WhatsApp RAG Project

A Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph for answering queries about C.V. Raman Global University, Bhubaneswar, Odisha, using WhatsApp chat data, PDFs, images, and web scraping.

## Project Overview

- **Purpose**: RAG pipeline with multi-modal support (text, images), guardrails, and conversational memory
- **Tech Stack**: LangChain, LangGraph, FastAPI, Qdrant (vector store), Redis (image docstore), PostgreSQL (checkpointer), Fireworks (embeddings), Cohere (reranking), Portkey (LLM gateway), NeMo Guardrails, Logfire (observability)
- **Python Version**: 3.13

## Project Structure

```
whatsapp-rag/
├── src/
│   ├── config.py              # Centralized pydantic-settings config (reads .env)
│   ├── observability.py       # Logfire setup
│   ├── embeddings/            # Fireworks dense + BM25 sparse embeddings
│   ├── generation/            # LLM and prompt management
│   ├── graph/                 # LangGraph nodes, checkpointer, agent workflow
│   ├── ingestion/             # PDF/image processing, web scraping
│   ├── retrieval/             # Hybrid retrieval, reranking, semantic cache
│   └── vector_store/          # Qdrant client setup
├── api/                       # FastAPI chat API (api/main.py) — CORS-enabled, runs standalone
├── frontend/                  # Static chat UI (frontend/index.html) — runs standalone, calls the API cross-origin
├── scripts/
│   ├── run_ingestion.py       # Ingest documents into Qdrant
│   ├── ingest_dynamic.py      # Dynamic web scraping ingestion
│   ├── index_images.py        # Index images separately
│   ├── run_api.py             # Start the FastAPI chat API (Windows-safe event loop)
│   └── run_frontend.py        # Start the static frontend's own dev server
├── config/
│   ├── config.yml             # NeMo Guardrails config
│   └── prompts.yml            # Guardrails prompts
├── data/                      # Source markdown/PDFs
├── tests/                     # Pytest suite
├── .env                       # Environment secrets (NEVER commit)
├── pyproject.toml             # UV project metadata
└── requirements.txt           # Pip dependencies
```

## Development Workflow

### Environment Setup

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies (using uv)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Running the Application

```bash
# Run ingestion pipeline
python scripts/run_ingestion.py

# Start the FastAPI chat API (NOT `uvicorn api.main:app` directly on
# Windows — see scripts/run_api.py's docstring: psycopg needs a
# SelectorEventLoop forced before uvicorn picks its own loop class)
uv run python -m scripts.run_api      # http://localhost:8000

# In a second terminal, start the frontend
uv run python -m scripts.run_frontend # http://localhost:5500
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Async tests use pytest-asyncio (configured in pyproject.toml)
```

## Configuration

### Environment Variables

All settings are managed through `src/config.py` using `pydantic-settings`. Required variables in `.env`:

**Vector Store & Cache**
- `QDRANT_URL`, `QDRANT_API_KEY`
- `REDIS_URL`
- `DATABASE_URL` (PostgreSQL for LangGraph checkpointer)

**API Keys**
- `FIREWORKS_API_KEY` (dense embeddings)
- `GOOGLE_API_KEY` (Gemini for PDF extraction)
- `COHERE_API_KEY` (reranking)
- `PORTKEY_API_KEY` (LLM gateway)
- `LOGFIRE_TOKEN` (observability)

**Model Configs**
- `FIREWORKS_EMBEDDING_MODEL` (default: qwen3-embedding-8b)
- `GEMINI_MODEL` (default: gemini-2.5-flash)
- `PORTKEY_RESPONSE_CONFIG`, `PORTKEY_WORKER_CONFIG`

See `src/config.py` for complete list and defaults.

### Usage Pattern

```python
from src.config import get_settings

settings = get_settings()  # Cached singleton
settings.qdrant_url        # Type-safe access
```

## Code Conventions

- **No direct `os.environ` calls**: Always use `get_settings()` for environment variables
- **Async by default**: Most I/O operations (Qdrant, Redis, LLM calls) are async
- **Type hints**: Use throughout for clarity
- **Pydantic models**: For data validation and serialization
- **Docstrings**: Explain WHY, not WHAT (especially for non-obvious logic)

## Key Components

### Embeddings
- **Dense**: Fireworks Qwen3 (1024 dimensions)
- **Sparse**: Local BM25 via fastembed (no API key needed)
- **Hybrid retrieval**: Combines both for better recall

### Vector Store
- **Qdrant**: Two collections (`my_documents` for text, `image_documents` for images)
- **Indexing**: Run `scripts/run_ingestion.py` to populate

### LLM Routing
- **Portkey**: Gateway for response generation with fallback/load balancing
- **Models**: Configured via Portkey config IDs

### Guardrails
- **NeMo Guardrails**: Input/output validation, PII filtering, jailbreak prevention
- **Config**: `config/config.yml`, `config/prompts.yml`

### Observability
- **Logfire**: Traces all LangChain/LangGraph operations, API calls, timing
- **Environment**: Set `LOGFIRE_ENVIRONMENT=development|staging|production`

## Safety & Secrets

- **NEVER commit `.env`**: Already in `.gitignore`
- **API keys**: Only in `.env`, accessed via `get_settings()`
- **Credentials in code**: Use placeholders in examples, never hardcode

## Testing Notes

- Pytest async mode is auto-enabled (`asyncio_mode = "auto"`)
- Fixtures in `conftest.py`
- Mock external services (Qdrant, Redis, LLM calls) for unit tests
- Integration tests should use test collections/databases

## Common Tasks

### Add a new LLM provider
1. Add API key to `.env`
2. Add field to `Settings` in `src/config.py`
3. Create wrapper in `src/generation/llm.py`

### Add a new retrieval strategy
1. Implement in `src/retrieval/`
2. Update graph node in `src/graph/nodes/`
3. Add tests in `tests/retrieval/`

### Update embeddings model
1. Change `FIREWORKS_EMBEDDING_MODEL` in `.env`
2. Update `embedding_dimensions` if dimensions change
3. Re-run ingestion to rebuild index

## Data Sources

- **Static**: `data/cgu.md`, `data/fee_structure.md`
- **Dynamic**: Notice board, examination cell (scraped via scripts)
- **PDFs**: Extracted using Gemini API for markdown conversion
- **Images**: Indexed separately with multimodal embeddings

## Chat API & Frontend

Two independent services, not a monolith — the API has no knowledge of the frontend and vice versa, only a CORS allow-list connects them.

- **API** (`api/main.py`, port 8000): exposes `POST /api/chat` (`{message, session_id}` → `{reply, session_id}`) wired directly to the LangGraph agent, plus `GET /api/health`. `FRONTEND_ORIGINS` in that file is the CORS allow-list — add to it if the frontend is ever served from somewhere other than `scripts/run_frontend.py`'s default.
- **Frontend** (`frontend/index.html`, port 5500 via `scripts/run_frontend.py`): a single-file vanilla JS chat UI with no build step. `API_BASE` at the top of its `<script>` points at the API's URL — update it if the API moves (e.g. once deployed). It generates a `session_id` client-side on first use and keeps it in `localStorage`, so a returning browser tab continues the same conversation.
- Each `session_id` is used as the LangGraph `thread_id`, giving every browser session persistent, per-session conversation memory in Postgres.
- Run locally: `uv run python -m scripts.run_api` and, separately, `uv run python -m scripts.run_frontend`, then open `http://localhost:5500/`.

## Known Issues / TODOs

- Image retrieval (`get_college_images` in `src/retrieval/tools.py`) is now wired end-to-end: it embeds the query, matches it against the 5 indexed category captions in the `image_documents` Qdrant collection, looks up that category's file paths in the Redis docstore, and returns them as `/media/...` URLs served by a `StaticFiles` mount in `api/main.py` (`IMAGES_DIR` = the project's `images/` folder). Note this matches category *captions* with text embeddings, not true multimodal (e.g. CLIP) image embeddings — good enough for 5 fixed categories, but won't scale to open-ended per-image search.
- Consider adding rate limiting for LLM calls
- Add more comprehensive error handling in graph nodes
- No semantic response caching yet — `src/generation/llm.py` explicitly processes every query fresh; Redis is only used as the image docstore (`src/vector_store/redis_store.py`). A `RedisSemanticCache` was planned but never wired in.

---

**Last Updated**: 2026-08-14
