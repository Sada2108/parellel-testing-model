"""PDF extraction step.

Uses Unstructured's ``partition_pdf`` to turn a PDF file into a list of raw
content ``Element`` objects, configured to extract embedded images and infer
table structure for downstream multimodal processing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from unstructured.partition.pdf import partition_pdf

from config.settings import (
    PDF_EXTRACT_IMAGE_BLOCK_TYPES,
    PDF_INFER_TABLE_STRUCTURE,
    PDF_STRATEGY,
)
from src.logger import get_logger

logger = get_logger(__name__)


def partition_document(file_path: str | Path) -> list:
    """Extract structured content elements from a PDF.

    Args:
        file_path: Path to the PDF to process.

    Returns:
        A list of Unstructured ``Element`` objects (headings, narrative text,
        tables and extracted images).
    """
    pdf_path = Path(file_path)
    logger.info(
        "Extracting PDF elements: %s (strategy=%s, extract_images=%s, infer_tables=%s)",
        pdf_path.name,
        PDF_STRATEGY,
        PDF_EXTRACT_IMAGE_BLOCK_TYPES,
        PDF_INFER_TABLE_STRUCTURE,
    )

    # ``unstructured`` is verbose; keep its log noise out of our logs.
    logging.getLogger("unstructured").setLevel(logging.WARNING)

    return partition_pdf(
        filename=str(pdf_path),
        strategy=PDF_STRATEGY,
        extract_image_block_types=PDF_EXTRACT_IMAGE_BLOCK_TYPES,
        infer_table_structure=PDF_INFER_TABLE_STRUCTURE,
        extract_images_in_pdf=False,
        chunking_strategy=None,
    )
