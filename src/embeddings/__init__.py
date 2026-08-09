"""
Embeddings package: dense (Fireworks Qwen3) + sparse (local BM25).

Both are exposed as cached factory functions (`get_dense_embeddings`,
`get_sparse_embeddings`), not raw instances — always call the function,
don't instantiate the classes directly elsewhere in the codebase, or
you'll lose the single-instance-per-process guarantee.
"""

from src.embeddings.fireworks_embeddings import get_dense_embeddings
from src.embeddings.sparse_embeddings import get_sparse_embeddings

__all__ = ["get_dense_embeddings", "get_sparse_embeddings"]