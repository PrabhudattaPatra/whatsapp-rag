"""
Entrypoint script: scrape dynamic web sources (Notice Board & Examination Cell),
parse PDFs with Gemini 2.5 Flash, and index into Qdrant text vector store.

Run:
    python -m scripts.ingest_dynamic
"""

import sys
import logging
from uuid import uuid4

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import logfire

from src.observability import configure_observability
from src.ingestion.web_scraper import (
    scrape_notice_board,
    scrape_examination_cell,
    build_documents_from_pdfs,
)
from src.vector_store import get_text_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@logfire.instrument("ingest_dynamic_documents")
def ingest_dynamic_documents() -> dict[str, int]:
    """
    Scrape Notice Board and Examination Cell pages, parse PDFs into markdown
    using Gemini 2.5 Flash, and write Documents into Qdrant text collection.
    """
    vector_store = get_text_vector_store()

    # 1. Scrape and process Notice Board
    logfire.info("Scraping Notice Board listing...")
    notice_records = scrape_notice_board()
    notice_docs = build_documents_from_pdfs(notice_records, doc_type="notice_board")

    if notice_docs:
        notice_uuids = [str(uuid4()) for _ in range(len(notice_docs))]
        vector_store.add_documents(documents=notice_docs, ids=notice_uuids)

    # 2. Scrape and process Examination Cell
    logfire.info("Scraping Examination Cell listing...")
    exam_records = scrape_examination_cell()
    exam_docs = build_documents_from_pdfs(exam_records, doc_type="examination_cell")

    if exam_docs:
        exam_uuids = [str(uuid4()) for _ in range(len(exam_docs))]
        vector_store.add_documents(documents=exam_docs, ids=exam_uuids)

    results = {
        "notice_board": len(notice_docs),
        "examination_cell": len(exam_docs),
    }

    logfire.info("Dynamic document ingestion complete", results=results)
    return results


if __name__ == "__main__":
    configure_observability()
    results = ingest_dynamic_documents()
    print(f"\n✅ Dynamic Ingestion Complete:")
    print(f"  - Notice Board: {results['notice_board']} documents")
    print(f"  - Examination Cell: {results['examination_cell']} documents")
    print(f"Total: {sum(results.values())} documents ingested into Qdrant.")
