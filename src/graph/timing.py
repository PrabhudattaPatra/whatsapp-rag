"""
Lightweight per-node wall-clock timing for the LangGraph workflow.

Why this exists, separate from the `@logfire.instrument(...)` decorators
already on every node:
    Those spans are only inspectable in a configured Logfire project.
    `time_node` gives an immediate, always-visible number (printed to
    stdout and also logged via logfire.info) for local latency debugging,
    with zero knowledge of what's inside the node it wraps — it just
    brackets the call with a stopwatch. That's intentional: it lets us
    measure real per-node wall-clock time (including whatever guardrails,
    LLM, or retrieval calls a node makes) without reaching into or
    modifying any node's internals — in particular, src/graph/nodes/agent.py's
    NeMo Guardrails wiring is left completely untouched.

Node wall-clock time still won't include LangGraph's own per-step
checkpoint writes (those happen in the Pregel loop between nodes, not
inside any node function) — subtract the sum of per-node times from the
total graph.ainvoke() time to see that overhead separately.
"""

import functools
import inspect
import time

import logfire


def time_node(name: str):
    """Decorator: log a graph node function's wall-clock duration."""

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - start
                    logfire.info(f"⏱️ [Timing] {name} took {elapsed:.2f}s", node=name, duration_s=elapsed)
                    print(f"[TIMING] {name}: {elapsed:.2f}s")

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logfire.info(f"⏱️ [Timing] {name} took {elapsed:.2f}s", node=name, duration_s=elapsed)
                print(f"[TIMING] {name}: {elapsed:.2f}s")

        return sync_wrapper

    return decorator
