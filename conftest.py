"""
conftest.py — project-wide pytest configuration.

Why this file exists:
    On Windows, Python 3.8+ defaults to ProactorEventLoop, but psycopg3's
    async driver (used by AsyncPostgresSaver) requires SelectorEventLoop.

    This policy switch MUST happen at import time — before pytest-asyncio
    creates any event loop — which is why it lives here at module level
    rather than inside a fixture.
"""

import sys
import asyncio

if sys.platform == "win32":
    # Switch to WindowsSelectorEventLoopPolicy globally so every loop
    # created by pytest-asyncio (and anywhere else in this process)
    # is a SelectorEventLoop, making psycopg3 async connections work.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
