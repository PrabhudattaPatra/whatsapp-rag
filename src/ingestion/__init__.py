"""
Ingestion package: turns raw sources (markdown files, scraped web pages,
PDFs, image folders) into LangChain Documents ready for embedding +
storage.
"""

from src.ingestion.image_indexer import index_images
from src.ingestion.markdown_loader import (
    load_fee_structure_documents,
    load_faq_documents,
)
from src.ingestion.web_scraper import (
    build_documents_from_pdfs,
    scrape_examination_cell,
    scrape_notice_board,
)

__all__ = [
    "load_faq_documents",
    "load_fee_structure_documents",
    "scrape_notice_board",
    "scrape_examination_cell",
    "build_documents_from_pdfs",
    "index_images",
]