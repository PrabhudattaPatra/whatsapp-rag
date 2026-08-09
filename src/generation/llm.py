"""
LLM factories for the RAG application.

This module provides cached, logfire-instrumented factory functions to construct:
1. Response LLM (routes queries to the responder config via Portkey)
2. Grader/Worker LLM (routes queries to the worker config via Portkey)

No semantic caching is applied - all queries are processed fresh.
"""

from functools import lru_cache
import logfire
from langchain_openai import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from src.config import get_settings

settings = get_settings()


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
