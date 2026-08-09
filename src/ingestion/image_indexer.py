"""
Index campus image categories: one Document (caption) per category goes
into the Qdrant image vector store; the actual file paths for that
category go into the Redis docstore, keyed by the same doc_id.

This is a "run once to (re)populate storage" script, not something called
per-request — that's why it lives in ingestion/, not retrieval/.

Why this differs from the notebook:
    - `academic_year` and `is_active` were repeated identically in every
      one of the 5 category dicts. Pulled out to settings.academic_year
      / a single default, since duplicating the same value 5 times is
      exactly the kind of thing that goes stale (someone updates one
      category for the new academic year and forgets the other four).
    - Wrapped in @logfire.instrument since this does real I/O (Qdrant
      writes, Redis writes) and you'll want to know how long a full
      re-index takes and whether any category failed.
"""

import glob
import json
import os
import uuid

import logfire
from langchain_core.documents import Document

from src.config import get_settings
from src.vector_store import get_image_docstore, get_image_vector_store
from src.vector_store.redis_store import IMAGE_DOC_ID_KEY

settings = get_settings()

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

# Caption -> folder/category. This is indexing-time data, not app config
# (it only matters when you run this script), so it stays here rather
# than in Settings.
IMAGE_CATEGORIES: dict[str, dict[str, str]] = {
    "Classroom with desks, whiteboard, and chairs": {
        "folder": "images/classroom",
        "category": "classroom",
    },
    "Canteen with tables, chairs, and food counter": {
        "folder": "images/canteen",
        "category": "canteen",
    },
    "Campus grounds, buildings, and outdoor areas of CVRGU": {
        "folder": "images/campus",
        "category": "campus",
    },
    "Library with bookshelves, reading areas, and study desks": {
        "folder": "images/library",
        "category": "library",
    },
    "Hostel rooms, common areas, and hostel building": {
        "folder": "images/hostel",
        "category": "hostel",
    },
}


def _load_image_paths(folder: str) -> list[str]:
    """Return all jpg/jpeg/png paths in a folder."""
    paths: list[str] = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, pattern)))
    return paths


@logfire.instrument("index_images")
def index_images() -> int:
    """
    (Re)index all image categories into Qdrant + Redis.

    Idempotent-ish: re-running assigns fresh doc_ids and re-adds
    documents each time. If you need true idempotency (skip categories
    already indexed), that's a natural next enhancement — flag it if you
    want to add it now rather than later.

    Returns:
        Number of categories indexed.
    """
    vector_store = get_image_vector_store()
    docstore = get_image_docstore()

    summary_docs: list[Document] = []
    docstore_entries: list[tuple[str, bytes]] = []

    for caption, info in IMAGE_CATEGORIES.items():
        image_paths = _load_image_paths(info["folder"])
        if not image_paths:
            logfire.warn("No images found for category", category=info["category"], folder=info["folder"])
            continue

        doc_id = str(uuid.uuid4())
        metadata = {
            IMAGE_DOC_ID_KEY: doc_id,
            "category": info["category"],
            "academic_year": settings.academic_year,
            "is_active": True,
            "image_count": len(image_paths),
        }

        summary_docs.append(Document(page_content=caption, metadata=metadata))
        docstore_entries.append((doc_id, json.dumps(image_paths).encode("utf-8")))

    if not summary_docs:
        logfire.warn("No image categories had any images — nothing indexed")
        return 0

    vector_store.add_documents(summary_docs)
    docstore.mset(docstore_entries)

    logfire.info("Indexed image categories", count=len(summary_docs))
    return len(summary_docs)


if __name__ == "__main__":
    index_images()