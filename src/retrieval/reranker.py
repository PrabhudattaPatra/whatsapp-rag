# src/retrieval/reranker.py
import time
import logfire
from typing import List
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
from src.config import get_settings

_reranker: CohereRerank | None = None

def _get_reranker() -> CohereRerank:
    global _reranker
    if _reranker is None:
        logfire.info("🧠 Initializing Cohere Reranker...")
        settings = get_settings()
        _reranker = CohereRerank(model=settings.rerank_model, top_n=settings.rerank_top_n)
    return _reranker

def rerank_documents(query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to Cohere...", query=query[:100])

    try:
        reranker = _get_reranker()
        with logfire.span("Cohere Reranking API Call"):
            result = reranker.compress_documents(query=query, documents=documents)
        
        duration = time.time() - start_time
        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s. Returned {len(result)} docs.")
        return result
    except Exception as e:
        logfire.error("❌ [Reranker] Failed. Falling back to original docs.", error=str(e))
        return documents[:top_n]