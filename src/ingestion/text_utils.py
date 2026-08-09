"""
Shared text-cleaning utility for ingestion.

Why this file exists:
    Your notebook defined `clean_text` identically in two separate cells
    (once for FAQ chunks, once for fee-structure chunks). Duplicated
    utility functions are a maintenance trap: if you ever need to fix or
    extend the cleaning logic, you'd have to remember to update it in
    every copy. One shared function, imported everywhere, fixes that.

Note on observability:
    We do NOT wrap this in @logfire.instrument(). This is a cheap,
    in-memory string operation with no network/DB call and effectively
    no failure mode worth tracing. Instrumenting trivial functions adds
    noise to your dashboard without adding insight — save spans for
    things with real latency or real failure modes (API calls, DB
    queries, LLM calls). You'll see that distinction applied in
    pdf_extractor.py and web_scraper.py.
"""


def clean_text(text: str) -> str:
    """Strip invalid surrogate characters that break UTF-8 encoding downstream."""
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")