"""
Observability bootstrap for the CGU RAG application, using Pydantic Logfire.

Why this file exists:
    We want automatic timing + tracing on the expensive parts of the RAG
    pipeline (embedding calls, Qdrant queries, Cohere reranking, LLM calls,
    the LangGraph nodes) without littering every function with manual
    `time.time()` calls and print statements.

    Logfire gives us:
      - `@logfire.instrument("name")` — wraps a function in a timed span,
        automatically records exceptions with full context, and shows up
        in the Logfire dashboard.
      - `with logfire.span("name"):` — same idea for a block of code inside
        a function, when you want finer-grained timing than the whole
        function (e.g. just the Qdrant call inside a bigger retrieval fn).
      - Structured logging: `logfire.info("retrieved {n} docs", n=len(docs))`
        instead of `print(f"retrieved {len(docs)} docs")` — queryable later,
        not just text in a terminal.

Usage:
    Call `configure_observability()` ONCE, at application startup
    (e.g. top of `api/main.py`, or top of `scripts/run_ingestion.py`).
    Every other module just does `import logfire` and uses
    `@logfire.instrument(...)` / `logfire.span(...)` directly — they don't
    need to configure anything themselves.
"""

import logfire

from src.config import get_settings


def configure_observability() -> None:
    """
    Configure Logfire for this process.

    Call this exactly once, as early as possible in your entrypoint
    (API startup, or the top of a standalone script). Calling it multiple
    times is harmless but wasteful — that's why entrypoints should call
    it, not every module.
    """
    settings = get_settings()

    logfire.configure(
        token=settings.logfire_token,
        service_name=settings.logfire_service_name,
        environment=settings.logfire_environment,
    )

    # Auto-instrument common libraries you're already using, so their calls
    # (HTTP requests, Postgres queries via the checkpointer, etc.) show up
    # as spans automatically, without decorating every call site by hand.
    logfire.instrument_httpx()
    logfire.instrument_requests()