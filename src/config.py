"""
Central configuration for the CGU RAG application.

Why this file exists:
    In the notebook, every module called `os.environ["SOME_KEY"]` directly.
    That means:
      - A typo in a key name fails at runtime, deep inside some function,
        instead of at startup.
      - There's no single place to see every setting the app needs.
      - You can't validate types (e.g. is REDIS_TTL actually an int?).

    `Settings` below fixes all three: it reads `.env` once, validates every
    field, and gives you `settings.qdrant_url` instead of
    `os.environ["QDRANT_URL"]` everywhere.

Usage:
    from src.config import get_settings
    settings = get_settings()
    settings.qdrant_url
"""

from functools import lru_cache
from dotenv import load_dotenv

# Load and populate os.environ so that third-party SDKs (NeMo Guardrails, etc.) can access keys
load_dotenv(override=True)

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All environment-driven configuration for the app, in one typed place."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars instead of erroring
    )

    # --- Fireworks (dense embeddings) ---
    fireworks_api_key: str
    fireworks_embedding_model: str = "accounts/fireworks/models/qwen3-embedding-8b"
    embedding_dimensions: int = 1024

    # --- Sparse embeddings (BM25 via fastembed, runs locally, no API key) ---
    sparse_embedding_model: str = "Qdrant/bm25"

    # --- Qdrant (vector store) ---
    qdrant_url: str
    qdrant_api_key: str
    qdrant_text_collection: str = "my_documents"
    qdrant_image_collection: str = "image_documents"

    # --- Redis (semantic cache + image docstore) ---
    redis_url: str
    cache_distance_threshold: float = 0.2
    cache_ttl_seconds: int = 3600

    # --- Postgres (LangGraph checkpointer / conversation memory) ---
    database_url: str

    # --- Google Gemini (PDF -> markdown extraction) ---
    google_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # --- Cohere (reranker) ---
    cohere_api_key: str
    rerank_model: str = "rerank-english-v3.0"
    rerank_top_n: int = 3

    # --- Portkey gateway (response LLM routing) ---
    portkey_api_key: str
    portkey_response_config: str = "pc-respon-d7d917"
    portkey_worker_config: str = "pc-worker-acd39e"

    # --- NeMo Guardrails ---
    guardrails_config_path: str = "./config"

    # --- Logfire (observability: tracing, timing, error capture) ---
    logfire_token: str
    logfire_service_name: str = "cgu-rag"
    logfire_environment: str = "development"  # "development" | "staging" | "production"

    # --- App-level constants ---
    academic_year: str = "2026-27"
    notice_board_lookback_days: int = 60
    examination_lookback_days: int = 30

    # --- Data source paths & URLs ---
    faq_data_path: str = "data/cgu.md"
    fee_structure_data_path: str = "data/fee_structure.md"
    notice_board_url: str = "https://cgu-odisha.ac.in/notice/"
    examination_cell_url: str = "https://cgu-odisha.ac.in/examination-cell/"


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    lru_cache means Settings() only runs once per process — the .env file
    is only read and validated the first time this is called, and every
    subsequent call reuses the same object. This also means: if required
    env vars are missing, your app fails immediately on first access,
    not silently mid-request.
    """
    return Settings()