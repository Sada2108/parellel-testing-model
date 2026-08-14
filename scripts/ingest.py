"""Ingestion pipeline entry point.

Partitions a PDF, chunks it, enriches it with AI summaries, exports to JSON
and creates a Chroma vector store, then runs a quick retrieval smoke test.

Usage:
    python -m scripts.ingest                          # default path
    python -m scripts.ingest path/to/document.pdf     # custom path
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.chunk import create_chunks_by_title
from src.ingestion.enrich import summarise_chunks
from src.ingestion.export import export_chunks_to_json
from src.ingestion.extract import partition_document
from src.ingestion.pipeline import create_vector_store
from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PDF_PATH = "./data/datasheet.pdf"


def run_with_retrieval_test(pdf_path: str) -> None:
    """Run the full pipeline, then do a quick retrieval test."""
    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    summarised = summarise_chunks(chunks)

    export_chunks_to_json(summarised, filename="chunks_huggingface.json")
    db = create_vector_store(summarised)

    query = "explain the internal architecture"
    retriever = db.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke(query)

    export_chunks_to_json(results, "rag_results.json")
    logger.info("Done. Retrieved %d chunks for query: %s", len(results), query)


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH

    logger.info("=" * 60)
    logger.info("Multimodal RAG Ingestion Pipeline")
    logger.info("=" * 60)

    run_with_retrieval_test(pdf)
