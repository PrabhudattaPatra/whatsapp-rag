"""
LLM factories and caching configuration.

This module provides cached, logfire-instrumented factory functions to construct:
1. Response LLM (routes queries to the responder config via Portkey)
2. Grader/Worker LLM (routes queries to the worker config via Portkey)

It also configures the RedisSemanticCache for LLM caching.
"""

from functools import lru_cache
import logfire
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_llm_cache
from langchain_redis import RedisSemanticCache
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from src.config import get_settings
from src.embeddings import get_dense_embeddings

settings = get_settings()

# --- Configure LLM Cache ---
try:
    logfire.info("Setting up RedisSemanticCache for LLM...", redis_url=settings.redis_url)
    dense_embeddings = get_dense_embeddings()
    set_llm_cache(
        RedisSemanticCache(
            embeddings=dense_embeddings,
            redis_url=settings.redis_url,
            distance_threshold=settings.cache_distance_threshold,
            ttl=settings.cache_ttl_seconds,
        )
    )
    logfire.info("RedisSemanticCache set up successfully.")
except Exception as e:
    logfire.error("Failed to initialize RedisSemanticCache", error=str(e))


@lru_cache
def get_response_llm() -> ChatOpenAI:
    """
    Cached response LLM factory using Portkey gateway response config.
    """
    logfire.info("Initializing response LLM via Portkey gateway...")
    headers = createHeaders(
        api_key=settings.portkey_api_key,
        config=settings.portkey_response_config,
    )
    return ChatOpenAI(
        api_key="X",
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        temperature=0.0,
    )


@lru_cache
def get_grader_llm() -> ChatOpenAI:
    """
    Cached grader/worker LLM factory using Portkey gateway worker config.
    Used for document grading and query classification.
    """
    logfire.info("Initializing grader/worker LLM via Portkey gateway...")
    headers = createHeaders(
        api_key=settings.portkey_api_key,
        config=settings.portkey_worker_config,
    )
    return ChatOpenAI(
        api_key="X",
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        temperature=0.0,
    )
