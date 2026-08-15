# WhatsApp RAG — CGU Assistant

A Retrieval-Augmented Generation (RAG) chatbot for **C. V. Raman Global University (CGU/CVRGU), Bhubaneswar, Odisha** — answers questions about admissions, academics, fees, exams, hostel, and campus life, grounded in the university's own fee structure, FAQ, notice board, and examination cell content. Built with LangGraph, FastAPI, and Qdrant, with guardrails, conversational memory, image retrieval, and an automated evaluation pipeline.

## Features

- **Hybrid retrieval** — dense (Fireworks Qwen3 embeddings) + sparse (local BM25) search over Qdrant, reranked with Cohere, filtered on document type / academic year / active status.
- **Agentic graph** (LangGraph) — routes each turn through tool-calling, document grading, query rewriting, and answer generation, with persistent per-session memory via a Postgres checkpointer.
- **Image retrieval** — matches a query against 5 indexed campus-image categories (classroom, canteen, campus, library, hostel) and serves the actual files.
- **Guardrails** — NeMo Guardrails input/output rails (jailbreak detection, content safety, topic control) run **in parallel** for lower latency, with a history-aware topic classifier so short follow-up questions ("give me for btech" after an M.Tech question) aren't misjudged as off-topic.
- **Streaming chat API + standalone frontend** — FastAPI backend streams tokens over SSE; a single-file vanilla-JS frontend renders them live.
- **Automated evaluation** — a LangSmith-based pipeline drives the real production graph end-to-end and scores answers on correctness, groundedness, relevance, and retrieval relevance.
- **Dynamic ingestion** — scheduled/on-demand scraping of the notice board and examination cell, with PDF-to-markdown extraction via Gemini.
- **Observability** — every graph node, retrieval call, and LLM request is traced with Logfire.

## Architecture

```
User message
   │
   ▼
┌─────────────┐   tool call?   ┌──────────┐   relevant?   ┌──────────┐
│   agent     │ ─────────────▶ │ retrieve │ ────────────▶ │ generate │ ──▶ END
│ (guardrails │                │ (Qdrant +│                └──────────┘
│  + routing) │                │  Cohere) │       │ not relevant
└─────────────┘ ◀───────────── └──────────┘       ▼
      ▲          no tool call          ┌──────────┐
      │                                │ rewrite  │
      └────────────────────────────────┘ question │
                                        └──────────┘
```

- **`agent`** (`src/graph/nodes/agent.py`) — wraps the response LLM in NeMo Guardrails (`RunnableRails`), decides whether to answer directly or call a retrieval tool.
- **`retrieve`** — a LangGraph `ToolNode` running `classify_and_retrieve` (hybrid Qdrant search + Cohere rerank) or `get_college_images` (`src/retrieval/tools.py`).
- **`grade_documents`** — a conditional edge; fast-paths on the reranker's relevance score, falls back to an LLM grader.
- **`rewrite`** — reformulates the question and loops back to `agent` when retrieved documents aren't relevant.
- **`generate`** — streams the final answer from retrieved context (or returns a canned reply with image URLs for the image-tool path).

State (`src/graph/state.py`) persists per `thread_id` in Postgres, so a browser session's conversation history survives across turns.

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain, LangGraph |
| API | FastAPI (SSE streaming), Uvicorn |
| Vector store | Qdrant (hybrid dense+sparse, TurboQuant 4-bit quantization on the text collection) |
| Embeddings | Fireworks (Qwen3, dense) + local BM25 via fastembed (sparse) |
| Reranking | Cohere |
| LLM gateway | Portkey (response + worker/grader model routing) |
| Guardrails | NeMo Guardrails (NVIDIA NIM content-safety/jailbreak models, Groq topic classifier) |
| Conversation memory | PostgreSQL (`AsyncPostgresSaver`) |
| Image docstore | Redis |
| PDF extraction | Google Gemini |
| Observability | Logfire |
| Evaluation | LangSmith (`aevaluate`, LLM-as-judge evaluators) |
| Frontend | Vanilla JS, single HTML file, no build step |

## Project structure

```
whatsapp-rag/
├── src/
│   ├── config.py              # Centralized pydantic-settings config (reads .env)
│   ├── observability.py       # Logfire setup
│   ├── embeddings/            # Fireworks dense + BM25 sparse embeddings
│   ├── generation/             # LLM factories (Portkey) and prompts
│   ├── graph/                 # LangGraph state, nodes, workflow, checkpointer
│   ├── ingestion/              # Markdown/PDF loading, web scraping, image indexing
│   ├── retrieval/              # Hybrid retrieval + rerank tools
│   └── vector_store/            # Qdrant client + collection setup
├── api/                       # FastAPI chat API (api/main.py)
├── frontend/                   # Static chat UI (frontend/index.html)
├── evaluation/                 # LangSmith evaluation pipeline (dataset, target, evaluators)
├── config/                     # NeMo Guardrails config (config.yml, prompts.yml, actions.py)
├── scripts/                     # Entrypoints: ingestion, image indexing, run API/frontend
├── data/                       # Source markdown (fee structure, FAQ)
├── images/                     # Campus photos, grouped by category
├── tests/                       # Pytest suite
├── .env                         # Environment secrets (never commit)
├── pyproject.toml               # uv project metadata
└── requirements.txt              # pip fallback
```

## Setup

### Prerequisites

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- A Qdrant instance (cloud or self-hosted), a Postgres database, and a Redis instance

### Install

```bash
uv sync
# or: pip install -r requirements.txt
```

### Configure `.env`

All settings are typed and validated in `src/config.py` (`get_settings()`). Required variables:

| Variable | Purpose |
|---|---|
| `QDRANT_URL`, `QDRANT_API_KEY` | Vector store |
| `REDIS_URL` | Image docstore |
| `DATABASE_URL` | Postgres (LangGraph checkpointer) |
| `FIREWORKS_API_KEY` | Dense embeddings |
| `GOOGLE_API_KEY` | Gemini (PDF → markdown extraction) |
| `COHERE_API_KEY` | Reranking |
| `GROQ_API_KEY` | Query classifier (and guardrails topic check) |
| `PORTKEY_API_KEY` | LLM gateway (response + grader models) |
| `NVIDIA_API_KEY` | Guardrails NIM models (jailbreak detection, content safety) — read directly by NeMo Guardrails, not through `src/config.py` |
| `LOGFIRE_TOKEN` | Observability |
| `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Optional — only needed to run `evaluation/` |

See `src/config.py` for the full list, defaults, and optional overrides (embedding model, rerank model, academic year, etc.).

### Ingest data

```bash
uv run python -m scripts.run_ingestion    # static: FAQ + fee structure (data/*.md)
uv run python -m scripts.ingest_dynamic   # dynamic: notice board + examination cell (scraped + Gemini-parsed PDFs)
uv run python -m scripts.index_images     # campus image categories → Qdrant + Redis
```

All three are idempotent — safe to re-run after updating `data/*.md` or when the source website changes.

### Run the app

```bash
# Terminal 1 — API (http://localhost:8000)
uv run python -m scripts.run_api

# Terminal 2 — frontend (http://localhost:5500)
uv run python -m scripts.run_frontend
```

Open `http://localhost:5500`. The frontend generates a `session_id` on first use (kept in `localStorage`) and sends it with every request; the API uses it as the LangGraph `thread_id`, so a returning tab continues the same conversation.

> On Windows, always use `scripts/run_api.py` rather than `uvicorn api.main:app` directly — psycopg needs a `SelectorEventLoop` forced before uvicorn picks its own loop class.

## Evaluation

`evaluation/` runs the real, compiled production graph against a labeled dataset and scores every answer with four LLM-as-judge evaluators (correctness, groundedness, relevance, retrieval relevance), using LangSmith's `aevaluate`.

```bash
PYTHONIOENCODING=utf-8 uv run python -m evaluation.run_evaluation
```

- Dataset (`evaluation/datasets.py`) is bootstrapped automatically in LangSmith on first run, and synced (missing examples added) on later runs.
- Judges reuse the project's own Portkey-routed `get_grader_llm()` — no separate OpenAI key needed.
- Results (per-example scores + explanations) are viewable in the LangSmith UI; a summary `pandas.DataFrame` is also printed to the console.

## Testing

```bash
pytest
pytest --cov=src tests/
```

Async tests run automatically (`asyncio_mode = "auto"`, configured in `pyproject.toml`). External services (Qdrant, Redis, LLMs) should be mocked in unit tests.

## Configuration notes

- **Guardrails run in parallel** (`config/config.yml` → `rails.input.parallel: true`) — jailbreak detection, content safety, and topic control fire concurrently instead of sequentially, cutting input-rail latency to the slowest single check instead of their sum.
- **`config/actions.py`** overrides NeMo's built-in `self_check_input` action to consider the immediately-preceding turn when judging topic relevance, so short follow-ups ("give me for btech") aren't misjudged as off-topic in isolation.
- **TurboQuant (4-bit) quantization** is enabled on the `my_documents` Qdrant collection's dense vectors (`src/vector_store/qdrant_client.py`) — ~8x storage reduction with minimal recall loss; the image collection is left unquantized.
- **Guardrails NIM/Groq calls** and most third-party API usage here run on free-tier quotas — expect occasional latency variance from those providers, independent of anything in this codebase.

## Known limitations

- Image retrieval matches query text against 5 fixed category *captions*, not true multimodal (e.g. CLIP) image embeddings — works well for the current 5 categories, won't scale to open-ended per-image search.
- No semantic response caching — every query is processed fresh (Redis is only used as the image docstore).
- The evaluation dataset covers fee-structure Q&A; it should be refreshed whenever `data/fee_structure.md` changes (ground-truth answers go stale otherwise — this happened once already).
