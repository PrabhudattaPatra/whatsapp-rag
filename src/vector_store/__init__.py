"""
Vector store package: Qdrant (text + image collections) and Redis
(image docstore + later, LLM semantic cache).

As with embeddings: always call the get_X() factory functions, never
instantiate QdrantClient / QdrantVectorStore / redis.Redis directly
elsewhere in the codebase.
"""

from src.vector_store.qdrant_client import (
    get_image_vector_store,
    get_qdrant_client,
    get_text_vector_store,
)
from src.vector_store.redis_store import get_image_docstore, get_redis_client

__all__ = [
    "get_qdrant_client",
    "get_text_vector_store",
    "get_image_vector_store",
    "get_redis_client",
    "get_image_docstore",
]