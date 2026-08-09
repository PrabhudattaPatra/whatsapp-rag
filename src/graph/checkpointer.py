"""
Async Postgres checkpointer for LangGraph conversation memory.

Why this file exists:
    AsyncPostgresSaver requires an async context manager to open and cleanly
    close its connection pool. In the notebook this was done inline with
    `await init_graph_with_memory()`. In production the lifecycle must be
    tied to the app startup/shutdown events so we don't leak connections.

    An `AsyncExitStack` is stored here at module level. The entrypoint
    (API lifespan or script) calls `get_checkpointer()` on startup to enter
    the context and `close_checkpointer()` on shutdown to exit it cleanly.

Usage:
    # In FastAPI lifespan or startup script:
    from src.graph.checkpointer import get_checkpointer, close_checkpointer

    checkpointer = await get_checkpointer()   # opens connection pool
    # ... run the app ...
    await close_checkpointer()                # closes connection pool cleanly
"""

import logfire
from contextlib import AsyncExitStack

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.config import get_settings

settings = get_settings()

# Module-level state: one connection pool per process
_exit_stack: AsyncExitStack | None = None
_checkpointer: AsyncPostgresSaver | None = None


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Initialize and return the AsyncPostgresSaver, opening its connection pool.

    Idempotent — safe to call multiple times; returns the cached instance
    after the first call.

    Call once at application startup, before the graph is compiled or
    any requests are served.
    """
    global _exit_stack, _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    logfire.info("Initializing AsyncPostgresSaver connection pool...")
    _exit_stack = AsyncExitStack()
    _checkpointer = await _exit_stack.enter_async_context(
        AsyncPostgresSaver.from_conn_string(settings.database_url)
    )

    # `setup()` creates the required tables (checkpoints, writes, migrations).
    # Safe to call on every startup — it is idempotent once tables exist.
    await _checkpointer.setup()
    logfire.info("AsyncPostgresSaver ready.")
    return _checkpointer


async def close_checkpointer() -> None:
    """
    Close the AsyncPostgresSaver connection pool cleanly.

    Call once at application shutdown (FastAPI lifespan teardown or
    script finally block).
    """
    global _exit_stack, _checkpointer
    if _exit_stack is not None:
        logfire.info("Closing AsyncPostgresSaver connection pool...")
        await _exit_stack.aclose()
        _exit_stack = None
        _checkpointer = None
        logfire.info("AsyncPostgresSaver closed.")
