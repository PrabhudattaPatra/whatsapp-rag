"""
Scrape CGU's notice board and examination cell pages for recent PDF links,
then extract each PDF into a metadata-tagged Document.

Why this file exists:
    Your notebook had `extract_notice_board_data` and
    `extract_examination_data` — genuinely identical logic, differing
    only in the URL and the lookback window. Same story for
    `build_documents_from_pdfs`, copy-pasted with a different doc_type.
    Both are merged into single parameterized functions here.
"""

from datetime import datetime, timedelta

import logfire
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from src.config import get_settings
from src.ingestion.pdf_extractor import PDFExtractionError, extract_pdf_markdown

settings = get_settings()


@logfire.instrument("scrape_listing_page")
def scrape_listing_page(url: str, lookback_days: int) -> list[dict]:
    """
    Scrape a CGU listing page (notice board or examination cell) for rows
    published within the last `lookback_days`.

    Returns:
        List of dicts: Serial_No, Title, Publish_Date, Date_Object, PDF_Link.
        Sorted newest first.
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    cutoff = datetime.now() - timedelta(days=lookback_days)

    table = soup.find("table")
    if not table:
        raise ValueError(f"No table found on listing page: {url}")

    records = []
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        sl_no = cols[0].get_text(strip=True)
        title = cols[1].get_text(strip=True)
        date_str = cols[2].get_text(strip=True)

        pdf_link = None
        link_tag = cols[3].find("a")
        if link_tag and link_tag.get("href"):
            pdf_link = link_tag["href"]
            if not pdf_link.startswith("http"):
                pdf_link = f"https://cgu-odisha.ac.in{pdf_link}"

        try:
            notice_date = datetime.strptime(date_str, "%B %d, %Y")
        except ValueError:
            continue

        if notice_date >= cutoff:
            records.append(
                {
                    "Serial_No": sl_no,
                    "Title": title,
                    "Publish_Date": date_str,
                    "Date_Object": notice_date,
                    "PDF_Link": pdf_link or "No PDF available",
                }
            )

    records.sort(key=lambda r: r["Date_Object"], reverse=True)
    return records


def scrape_notice_board() -> list[dict]:
    """Recent notice-board entries, from config-driven URL and lookback window."""
    return scrape_listing_page(
        url=settings.notice_board_url,
        lookback_days=settings.notice_board_lookback_days,
    )


def scrape_examination_cell() -> list[dict]:
    """Recent examination-cell entries, from config-driven URL and lookback window."""
    return scrape_listing_page(
        url=settings.examination_cell_url,
        lookback_days=settings.examination_lookback_days,
    )


@logfire.instrument("build_documents_from_pdfs")
def build_documents_from_pdfs(records: list[dict], doc_type: str) -> list[Document]:
    """
    Turn scraped listing records into Documents by extracting each PDF's
    contents via Gemini.

    Args:
        records: Output of scrape_notice_board() / scrape_examination_cell().
        doc_type: Metadata tag — "notice_board" or "examination_cell".

    Failed extractions are logged (with full context, via logfire.error)
    and skipped, rather than crashing the whole batch — one bad PDF
    shouldn't block ingesting the other 20.
    """
    documents = []

    for record in records:
        try:
            markdown_content = extract_pdf_markdown(record["PDF_Link"])
        except PDFExtractionError as e:
            logfire.error(
                "Skipping PDF that failed extraction",
                title=record["Title"],
                pdf_url=record["PDF_Link"],
                error=str(e),
            )
            continue

        documents.append(
            Document(
                page_content=markdown_content,
                metadata={
                    "source": record["Title"],
                    "doc_type": doc_type,
                    "date": record["Publish_Date"],
                    "pdf_url": record["PDF_Link"],
                    "is_active": True,
                    "academic_year": settings.academic_year,
                },
                id=record["Serial_No"],
            )
        )

    return documents