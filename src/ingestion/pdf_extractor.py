"""
PDF -> clean markdown extraction, via Google Gemini.

Why this file exists:
    The notebook created a fresh `genai.Client` at global scope, twice
    (once per copy-pasted cell). We cache it with `lru_cache` — same
    pattern as `get_settings()` — so it's built once per process and
    reused, instead of relying on notebook cell execution order.

    We also raise a specific exception (`PDFExtractionError`) instead of
    swallowing failures with `print(f"Failed: {e}")`. The caller
    (web_scraper.build_documents_from_pdfs) decides what to do with a
    failure — log it and skip, retry, whatever. A low-level extraction
    function should never make that policy decision itself.
"""

from functools import lru_cache

import logfire
import requests
from google import genai
from google.genai import types

from src.config import get_settings

settings = get_settings()


class PDFExtractionError(Exception):
    """Raised when a PDF can't be downloaded or Gemini fails to extract it."""


@lru_cache
def get_gemini_client() -> genai.Client:
    """Cached Gemini client — built once per process, not once per call."""
    return genai.Client(api_key=settings.google_api_key)


EXTRACTION_PROMPT = """
Extract this document into clean markdown.

Rules:
- Do NOT summarize.
- Do NOT omit information.
- Preserve all dates, names, URLs, fees, and codes.
- Preserve headings.
- Preserve tables as markdown tables.
- Preserve lists.
- Return markdown only.
"""


@logfire.instrument("extract_pdf_markdown", extract_args=True)
def extract_pdf_markdown(pdf_url: str) -> str:
    """
    Download a PDF and extract its contents as clean markdown via Gemini.

    Raises:
        PDFExtractionError: if the download or the Gemini call fails.
            Wrapped in @logfire.instrument, so the failure (with the
            original traceback) is automatically captured as a span
            error in the Logfire dashboard — no manual print() needed.
    """
    try:
        pdf_bytes = requests.get(pdf_url, timeout=30).content
    except requests.RequestException as e:
        raise PDFExtractionError(f"Failed to download PDF from {pdf_url}: {e}") from e

    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                EXTRACTION_PROMPT,
            ],
        )
    except Exception as e:
        raise PDFExtractionError(f"Gemini extraction failed for {pdf_url}: {e}") from e

    return response.text