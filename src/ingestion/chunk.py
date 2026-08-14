"""Title-based chunking of unstructured elements.

Groups the extracted PDF elements into semantic chunks that follow the
document's heading structure, so downstream steps (enrichment, embedding)
operate on coherent units of text.
"""

from __future__ import annotations

from unstructured.chunking.title import chunk_by_title

from config.settings import (
    CHUNK_COMBINE_UNDER_N_CHARS,
    CHUNK_ISOLATE_TABLES,
    CHUNK_MAX_CHARACTERS,
    CHUNK_NEW_AFTER_N_CHARS,
)
from src.logger import get_logger

logger = get_logger(__name__)


def create_chunks_by_title(elements: list) -> list:
    """Create intelligent chunks using a title-based strategy.

    Args:
        elements: Parsed PDF elements from :func:`partition_document`.

    Returns:
        List of unstructured chunks grouped by document heading.
    """
    logger.info("Creating smart chunks...")

    chunks = chunk_by_title(
        elements,
        max_characters=CHUNK_MAX_CHARACTERS,
        new_after_n_chars=CHUNK_NEW_AFTER_N_CHARS,
        combine_text_under_n_chars=CHUNK_COMBINE_UNDER_N_CHARS,
        isolate_table=CHUNK_ISOLATE_TABLES,
    )

    logger.info("Created %d chunks", len(chunks))
    return chunks
