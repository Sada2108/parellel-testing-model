"""Getting a datasheet into testable chunks -- upload or URL.

Reuses the pipeline's own extraction and chunking steps
(``src.ingestion.extract`` / ``src.ingestion.chunk``) so a chunk tested here
is built exactly the same way a chunk would be during real ingestion. Only
adds a second input path: fetch the PDF from a URL instead of requiring a
local upload first.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

from src.ingestion.chunk import create_chunks_by_title
from src.ingestion.enrich import separate_content_types
from src.ingestion.extract import partition_document
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TestChunk:
    chunk_id: str
    text: str
    tables: list[str] = field(default_factory=list)
    images_b64: list[str] = field(default_factory=list)


def fetch_pdf_from_url(url: str, timeout: int = 60) -> Path:
    """Download a datasheet PDF from a URL to a local temp file.

    Raises ``ValueError`` if the response doesn't look like a PDF, so a bad
    link fails loudly in the UI instead of silently feeding garbage into the
    extraction step.
    """
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    looks_like_pdf = resp.content[:5] == b"%PDF-" or "pdf" in content_type.lower()
    if not looks_like_pdf:
        raise ValueError(
            f"URL did not return a PDF (content-type={content_type!r}). "
            "Double-check the link points directly at the datasheet file."
        )

    suffix = ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(resp.content)
    tmp.close()
    logger.info("Downloaded datasheet from %s (%d bytes)", url, len(resp.content))
    return Path(tmp.name)


def load_chunks(pdf_path: str | Path, max_chunks: int | None = None) -> list[TestChunk]:
    """Partition + chunk a PDF and split each chunk into text/tables/images.

    Mirrors what ``src.ingestion.enrich.summarise_chunks`` does per chunk,
    minus the actual model call -- that part is what the test harness swaps
    in per candidate model.
    """
    try:
        elements = partition_document(pdf_path)
    except Exception as e:
        message = str(e).lower()
        if "tesseract" in message or "poppler" in message:
            raise RuntimeError(
                f"{e}\n\n"
                "The hi_res PDF strategy needs Tesseract OCR and Poppler installed "
                "as system binaries (pip alone doesn't provide them). On macOS, "
                "install both with:\n\n"
                "    brew install tesseract poppler\n\n"
                "then restart the app."
            ) from e
        raise
    raw_chunks = create_chunks_by_title(elements)

    if max_chunks is not None:
        raw_chunks = raw_chunks[:max_chunks]

    out: list[TestChunk] = []
    for i, chunk in enumerate(raw_chunks):
        content = separate_content_types(chunk)
        out.append(
            TestChunk(
                chunk_id=str(i),
                text=content["text"],
                tables=content["tables"],
                images_b64=content["images"],
            )
        )
    return out
