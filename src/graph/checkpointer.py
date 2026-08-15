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

Why an explicit psycopg_pool.AsyncConnectionPool instead of
AsyncPostgresSaver.from_conn_string():
    Despite the name, `from_conn_string()` does NOT create a pool — it opens
    exactly one bare AsyncConnection and holds it open for the entire process
    lifetime, with no keepalives and no health check. If that single
    connection goes idle long enough for the server (or a NAT/firewall in
    between) to silently drop it, the next query fails with
    `psycopg.OperationalError: consuming input failed: SSL connection has
    been closed unexpectedly`. Building a real AsyncConnectionPool and
    passing it to AsyncPostgresSaver(conn=...) instead gets us: multiple
    recyclable connections, TCP keepalives, and a `check` callback that
    pings a connection before handing it out so a dead one is replaced
    instead of surfacing as an error mid-request.

Usage:
    # In FastAPI lifespan or startup script:
    from src.graph.checkpointer import get_checkpointer, close_checkpointer

    checkpointer = await get_checkpointer()   # opens connection pool
    # ... run the app ...
    await close_checkpointer()                # closes connection pool cleanly
"""

import logfire
from contextlib import AsyncExitStack

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.config import get_settings

settings = get_settings()

# Module-level state: one connection pool per process
_exit_stack: AsyncExitStack | None = None
_checkpointer: AsyncPostgresSaver | None = None
_pool: AsyncConnectionPool | None = None


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Initialize and return the AsyncPostgresSaver, opening its connection pool.

    Idempotent — safe to call multiple times; returns the cached instance
    after the first call.

    Call once at application startup, before the graph is compiled or
    any requests are served.
    """
    global _exit_stack, _checkpointer, _pool

    if _checkpointer is not None:
        return _checkpointer

    logfire.info("Initializing AsyncPostgresSaver connection pool...")
    _exit_stack = AsyncExitStack()

    _pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={
            # Required by AsyncPostgresSaver — matches what
            # from_conn_string() used to pass to its single connection.
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            # TCP keepalives so a dead connection is noticed quickly
            # instead of sitting silently stale in the pool.
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
        # Ping a connection before handing it out; a dead one is discarded
        # and replaced instead of raising OperationalError mid-request.
        check=AsyncConnectionPool.check_connection,
        max_idle=300,       # recycle idle connections well before most managed-Postgres idle timeouts
        max_lifetime=1800,  # also recycle long-lived ones periodically, defense in depth
    )
    await _pool.open(wait=True)
    _exit_stack.push_async_callback(_pool.close)

    _checkpointer = AsyncPostgresSaver(conn=_pool)

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
    global _exit_stack, _checkpointer, _pool
    if _exit_stack is not None:
        logfire.info("Closing AsyncPostgresSaver connection pool...")
        await _exit_stack.aclose()
        _exit_stack = None
        _checkpointer = None
        _pool = None
        logfire.info("AsyncPostgresSaver closed.")
