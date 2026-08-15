"""
Qdrant client + collection setup + vector store factories.

Why this file exists:
    Your notebook created two collections ("my_documents" for text,
    "image_documents" for images) with copy-pasted
    "if not exists: create_collection(...)" blocks — differing only in
    whether sparse vectors are configured. We generalize that into one
    `ensure_collection` function.

    As with embeddings, everything here is exposed as cached factory
    functions (`get_qdrant_client`, `get_text_vector_store`,
    `get_image_vector_store`) — call the function, don't build these
    objects inline elsewhere in the codebase.
"""

from functools import lru_cache

import logfire
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance,
    SparseVectorParams,
    TurboQuantBitSize,
    TurboQuantization,
    TurboQuantQuantizationConfig,
    VectorParams,
)
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from src.config import get_settings
from src.embeddings import get_dense_embeddings, get_sparse_embeddings

settings = get_settings()


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Cached Qdrant client — one connection per process."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


@logfire.instrument("ensure_qdrant_collection", extract_args=True)
def ensure_collection(
    collection_name: str, with_sparse: bool, with_turboquant: bool = False
) -> None:
    """
    Create the collection if it doesn't already exist. Idempotent — safe
    to call on every startup, since it's a no-op once the collection exists.

    Args:
        collection_name: Name of the Qdrant collection.
        with_sparse: Whether to also configure a sparse ("sparse") vector
            field alongside the dense one — True for the hybrid text
            collection, False for the dense-only image collection.
        with_turboquant: Whether to enable TurboQuant (4-bit) quantization
            on the "dense" vector (Qdrant 1.18+). Only used for fresh
            `create_collection` calls -- it does NOT retroactively apply to
            an already-existing collection (Qdrant has no "recreate on
            config drift" behavior, and we don't want this function
            silently re-indexing production data on every startup). On
            `my_documents`, this was originally turned on out-of-band via
            a one-off `update_collection` call; this flag just makes a
            from-scratch recreation (fresh environment, disaster recovery)
            reproduce that instead of silently losing it. 4-bit chosen over
            2-bit/1-bit for best recall, since this collection feeds real
            answers about fees/admissions/exams.
    """
    client = get_qdrant_client()
    if client.collection_exists(collection_name):
        return

    sparse_vectors_config = None
    if with_sparse:
        sparse_vectors_config = {
            "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        }

    quantization_config = None
    if with_turboquant:
        quantization_config = TurboQuantization(
            turbo=TurboQuantQuantizationConfig(bits=TurboQuantBitSize.BITS4)
        )

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
                quantization_config=quantization_config,
            )
        },
        sparse_vectors_config=sparse_vectors_config,
    )

    # Create payload indexes for performance optimization
    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.is_active",
        field_schema=models.PayloadSchemaType.BOOL,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.academic_year",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    if with_sparse:
        # Text-specific metadata fields
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.doc_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.Doc_Title",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    logfire.info("Created Qdrant collection and payload indexes", collection_name=collection_name, with_sparse=with_sparse)


@lru_cache
def get_text_vector_store() -> QdrantVectorStore:
    """
    Hybrid (dense + sparse/BM25) vector store for text documents
    (faq, fee_structure, notice_board, examination_cell).
    """
    ensure_collection(settings.qdrant_text_collection, with_sparse=True, with_turboquant=True)
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_text_collection,
        embedding=get_dense_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


@lru_cache
def get_image_vector_store() -> QdrantVectorStore:
    """Dense-only vector store for image category captions."""
    ensure_collection(settings.qdrant_image_collection, with_sparse=False)
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_image_collection,
        embedding=get_dense_embeddings(),
        retrieval_mode=RetrievalMode.DENSE,
        vector_name="dense",
    )