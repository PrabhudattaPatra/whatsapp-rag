"""
Sparse (BM25) embeddings via fastembed — runs locally, no API key needed.

Why this file exists:
    Same `FastEmbedSparse` as your notebook, just: model name pulled from
    `settings.sparse_embedding_model` instead of hardcoded, and wrapped
    in a cached factory so it's initialized once per process rather than
    wherever it happened to appear in notebook cell order.

    No @logfire.instrument here: this runs locally (no network call), so
    there's no API latency or remote-failure mode worth tracing. If you
    ever notice the *first* call is slow (model weights downloading on
    first use), that's a one-time cold-start cost, not something you'd
    want cluttering every subsequent span.
"""

from functools import lru_cache

from langchain_qdrant import FastEmbedSparse

from src.config import get_settings

settings = get_settings()


@lru_cache
def get_sparse_embeddings() -> FastEmbedSparse:
    """Cached singleton — build the sparse embedder once per process."""
    return FastEmbedSparse(model_name=settings.sparse_embedding_model)