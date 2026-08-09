"""
Markdown document loading + header-based chunking.

Why this file exists:
    Your notebook had two near-identical blocks: one loading + chunking
    `data/cgu.md` (FAQ, split on "##"), and one loading + chunking
    `data/fee_structure.md` (split on "#" and "##"). Both then did the
    same metadata-enrichment + clean_text loop. We generalize that into
    one function, `load_and_chunk_markdown`, parameterized by file path,
    header levels, and doc_type — then two thin wrapper functions for
    your two actual documents.

    If you add a third markdown source next semester, you write one
    3-line wrapper function, not copy-paste a whole cell.
"""

from datetime import date

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.config import get_settings
from src.ingestion.text_utils import clean_text

settings = get_settings()


def load_and_chunk_markdown(
    file_path: str,
    headers_to_split_on: list[tuple[str, str]],
    doc_type: str,
    source: str = "cgu",
) -> list[Document]:
    """
    Load a markdown file, split it on the given header levels, and attach
    standard RAG metadata to every chunk.

    Args:
        file_path: Path to the .md file.
        headers_to_split_on: e.g. [("##", "Question Header")] — same
            format LangChain's MarkdownHeaderTextSplitter expects.
        doc_type: Category tag stored in metadata (e.g. "faq",
            "fee_structure") — this is what classify_and_retrieve later
            filters on.
        source: Metadata source tag. Defaults to "cgu".

    Returns:
        List of cleaned, metadata-enriched Document chunks.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    splits = splitter.split_text(docs[0].page_content)

    today = date.today().isoformat()
    for chunk_index, chunk in enumerate(splits):
        chunk.page_content = clean_text(chunk.page_content)
        chunk.metadata.update(
            {
                "doc_type": doc_type,
                "source": source,
                "date": today,
                "pdf_url": None,
                "academic_year": settings.academic_year,
                "chunk_index": chunk_index,
                "is_active": True,
            }
        )

    return splits


def load_faq_documents() -> list[Document]:
    """Load and chunk the FAQ / CGU markdown source (split on '##' Q&A headers)."""
    return load_and_chunk_markdown(
        file_path=settings.faq_data_path,
        headers_to_split_on=[("##", "Question Header")],
        doc_type="faq",
    )


def load_fee_structure_documents() -> list[Document]:
    """Load and chunk the fee-structure markdown source (split on '#' and '##')."""
    return load_and_chunk_markdown(
        file_path=settings.fee_structure_data_path,
        headers_to_split_on=[("#", "Doc_Title"), ("##", "Fee_Category")],
        doc_type="fee_structure",
    )