"""
Dense embeddings via Fireworks AI (Qwen3 embedding model), wrapped to
satisfy LangChain's `Embeddings` interface.

Why this file exists:
    Same class as your notebook, but:
      - Reads config from `settings`, not `os.environ["FIREWORKS_API_KEY"]`
        directly — so a missing key fails at app startup (Settings
        validation), not on the first embed call.
      - `embed_documents` is wrapped with @logfire.instrument, since this
        is a real network call to Fireworks — worth timing, and worth
        automatic error capture if the API call fails.
      - A module-level cached factory (`get_dense_embeddings`) replaces
        the notebook's pattern of instantiating `FireworksQwen3Embeddings(...)`
        inline wherever it was needed — one shared instance per process.
"""

from functools import lru_cache

import logfire
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from src.config import get_settings

settings = get_settings()

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksQwen3Embeddings(Embeddings):
    """LangChain-compatible embeddings client backed by Fireworks AI."""

    def __init__(self, model: str, dimensions: int):
        self.client = OpenAI(
            api_key=settings.fireworks_api_key,
            base_url=FIREWORKS_BASE_URL,
        )
        self.model = model
        self.dimensions = dimensions

    @logfire.instrument("fireworks_embed_documents", extract_args=True)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. This is the actual network call to Fireworks."""
        resp = self.client.embeddings.create(
            input=texts,
            model=self.model,
            dimensions=self.dimensions,
        )
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Not separately instrumented — it just delegates to
        embed_documents (already instrumented), so you'd get a
        redundant nested span with no new information.
        """
        return self.embed_documents([text])[0]


@lru_cache
def get_dense_embeddings() -> FireworksQwen3Embeddings:
    """Cached singleton — build the embeddings client once per process."""
    return FireworksQwen3Embeddings(
        model=settings.fireworks_embedding_model,
        dimensions=settings.embedding_dimensions,
    )