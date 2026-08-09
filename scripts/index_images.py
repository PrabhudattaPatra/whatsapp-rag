"""
Entrypoint script: index multimodal campus images into Qdrant & Redis.

Run via CLI:
    python -m scripts.index_images
"""

import sys
import logging

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import logfire
from src.observability import configure_observability
from src.ingestion.image_indexer import index_images

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@logfire.instrument("index_campus_images")
def main():
    """Main execution function to index all campus image categories."""
    configure_observability()
    logger.info("Starting campus image indexing process...")

    # Uses the default catalog & metadata mapping defined in image_indexer module
    indexed_count = index_images()

    logfire.info("Campus image indexing completed successfully.", total_categories=indexed_count)
    print(f"\n✅ Indexing complete: {indexed_count} image categories indexed into Qdrant & Redis.")


if __name__ == "__main__":
    main()
