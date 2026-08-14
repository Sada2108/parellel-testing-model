"""Ingestion package.

Exposes the full ingestion pipeline and its individual steps so callers can
either run everything (:func:`run_complete_ingestion_pipeline`) or invoke a
single stage in isolation.
"""

from src.ingestion.chunk import create_chunks_by_title
from src.ingestion.enrich import (
    create_ai_enhanced_summary,
    separate_content_types,
    summarise_chunks,
)
from src.ingestion.export import export_chunks_to_json
from src.ingestion.extract import partition_document
from src.ingestion.pipeline import create_vector_store, run_complete_ingestion_pipeline

__all__ = [
    "partition_document",
    "create_chunks_by_title",
    "separate_content_types",
    "create_ai_enhanced_summary",
    "summarise_chunks",
    "export_chunks_to_json",
    "create_vector_store",
    "run_complete_ingestion_pipeline",
]
