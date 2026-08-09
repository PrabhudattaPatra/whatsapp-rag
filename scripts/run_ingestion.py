"""
Entrypoint script: load documents into the Qdrant text vector store.

Run:
    python -m scripts.run_ingestion

Why this file exists:
    In the notebook, "run ingestion" meant re-running notebook cells in
    order. In production, ingestion is a deliberate, repeatable operation
    you trigger on demand (or on a schedule) — this script is that
    trigger point.

Note on structure:
    This currently ingests only the static markdown sources (FAQ, fee
    structure). Web-scraped sources (notice board, examination cell) and
    images are separate, independently-runnable pipelines — we'll wire
    them into this same script as separate functions once we've built
    and tested them, so you can run "just the static docs" or "just the
    notice board" without re-running everything.

This is also the first entrypoint in the project, which is why it's the
one that calls configure_observability() — library/module code never
configures Logfire itself, only entrypoints do (see src/observability.py).
"""

import logfire

from src.ingestion import load_fee_structure_documents, load_faq_documents
from src.observability import configure_observability
from src.vector_store import get_text_vector_store


@logfire.instrument("ingest_static_documents")
def ingest_static_documents() -> int:
    """
    Load the FAQ and fee-structure markdown sources, chunk them, and
    write them into the Qdrant text collection.

    Returns:
        Total number of chunks written.
    """
    vector_store = get_text_vector_store()

    faq_chunks = load_faq_documents()
    fee_chunks = load_fee_structure_documents()

    logfire.info(
        "Loaded static document chunks",
        faq_count=len(faq_chunks),
        fee_structure_count=len(fee_chunks),
    )

    vector_store.add_documents(faq_chunks)
    vector_store.add_documents(fee_chunks)

    total = len(faq_chunks) + len(fee_chunks)
    logfire.info("Static document ingestion complete", total_chunks=total)
    return total


if __name__ == "__main__":
    configure_observability()
    count = ingest_static_documents()
    print(f"Ingested {count} chunks (FAQ + fee structure) into Qdrant.")