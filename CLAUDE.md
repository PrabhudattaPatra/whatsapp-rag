# WhatsApp RAG Project

A Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph for answering queries about CGU (Centurion University of Technology and Management) using WhatsApp chat data, PDFs, images, and web scraping.

## Project Overview

- **Purpose**: RAG pipeline with multi-modal support (text, images), semantic caching, guardrails, and conversational memory
- **Tech Stack**: LangChain, LangGraph, FastAPI, Qdrant (vector store), Redis (cache), PostgreSQL (checkpointer), Fireworks (embeddings), Cohere (reranking), Portkey (LLM gateway), NeMo Guardrails, Logfire (observability)
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
├── api/                       # FastAPI endpoints (placeholder)
├── scripts/
│   ├── run_ingestion.py       # Ingest documents into Qdrant
│   ├── ingest_dynamic.py      # Dynamic web scraping ingestion
│   └── index_images.py        # Index images separately
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

# Start FastAPI server (when implemented)
uvicorn api.main:app --reload
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

### Semantic Cache
- **Redis**: Caches similar queries within 0.2 distance threshold
- **TTL**: 3600 seconds (1 hour)

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

## Known Issues / TODOs

- FastAPI endpoints in `api/` are placeholders
- Image retrieval needs multimodal embedding support fully integrated
- Consider adding rate limiting for LLM calls
- Add more comprehensive error handling in graph nodes

---

**Last Updated**: 2026-08-09
