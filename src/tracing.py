"""LangSmith trace metadata helpers.

Converts retrieved chunks into small, JSON-serialisable dictionaries so that
trace ``input``/``output`` metadata stays readable in the LangSmith UI instead
of dumping raw page content and base64 blobs.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.documents import Document

from src.documents import extract_original_data


def _extract_chunk_summary(chunk: Document) -> dict[str, Any]:
    """Build a compact summary of one chunk's multimodal content.

    The returned dict describes the chunk (lengths and presence of text,
    tables, images) rather than its full payload, which keeps trace metadata
    small and useful for debugging retrieval quality.
    """
    enhanced = chunk.page_content or ""
    original = extract_original_data(chunk)
    tables = original["tables_html"]
    images = original["images_base64"]

    return {
        "enhanced_length": len(enhanced),
        "enhanced_preview": enhanced[:300],
        "raw_text_length": len(original["raw_text"]),
        "table_count": len(tables),
        "image_count": len(images),
        "has_table": len(tables) > 0,
        "has_image": len(images) > 0,
    }


def summarize_chunks(chunks: list[Document], max_preview: int = 5) -> list[dict]:
    """Summarise up to ``max_preview`` chunks for inclusion in trace metadata.

    Args:
        chunks: Retrieved documents to describe.
        max_preview: Upper bound on the number of chunks summarised.

    Returns:
        List of compact chunk summaries (see :func:`_extract_chunk_summary`).
    """
    return [_extract_chunk_summary(c) for c in chunks[:max_preview]]
