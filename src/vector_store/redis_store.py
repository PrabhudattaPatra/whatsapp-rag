"""
Redis client + docstore factory.

Two separate uses of Redis in your notebook:
    1. `RedisStore` (via langchain_community) — maps an image-category
       doc_id -> JSON list of file paths. Used by the image retrieval tool.
    2. `RedisSemanticCache` — caches LLM responses. That belongs with the
       LLM setup, not here, so it'll live in src/generation/llm.py in a
       later step. This file only covers the docstore.
"""

from functools import lru_cache

import redis
from langchain_community.storage import RedisStore

from src.config import get_settings

settings = get_settings()

IMAGE_DOC_ID_KEY = "doc_id"


@lru_cache
def get_redis_client() -> redis.Redis:
    """Cached Redis client — one connection pool per process."""
    return redis.Redis.from_url(settings.redis_url)


@lru_cache
def get_image_docstore() -> RedisStore:
    """Cached docstore mapping image-category doc_id -> JSON file path list."""
    return RedisStore(client=get_redis_client())