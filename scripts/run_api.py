"""
Entry point for running the FastAPI chat API + frontend locally on Windows.

Why this script exists instead of `uvicorn api.main:app --reload` directly:
    psycopg (via AsyncPostgresSaver) requires a SelectorEventLoop. On
    Windows, uvicorn's `Server.run()` doesn't respect
    `asyncio.set_event_loop_policy()` at all — since uvicorn 0.36 it picks
    the loop class itself via a `loop_factory` (see
    `uvicorn/loops/asyncio.py`), which hardcodes `asyncio.ProactorEventLoop`
    for win32. So instead of calling `Server.run()` (or `uvicorn.run()`),
    this script drives `Server.serve()` directly through `asyncio.run(...,
    loop_factory=asyncio.SelectorEventLoop)`, forcing the Selector-based
    loop psycopg needs.

    --reload is intentionally not used here: uvicorn's reloader respawns
    worker processes in a way that isn't guaranteed to preserve this
    loop fix. Restart this script manually after code changes instead.

Usage:
    uv run python -m scripts.run_api
"""

import asyncio
import sys

import uvicorn


def main() -> None:
    config = uvicorn.Config("api.main:app", host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)

    if sys.platform == "win32":
        asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)
    else:
        server.run()


if __name__ == "__main__":
    main()
