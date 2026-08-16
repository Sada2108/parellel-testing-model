"""Getting a datasheet into testable chunks -- upload or URL.

Reuses the pipeline's own extraction and chunking steps
(``src.ingestion.extract`` / ``src.ingestion.chunk``) so a chunk tested here
is built exactly the same way a chunk would be during real ingestion. Only
adds a second input path: fetch the PDF from a URL instead of requiring a
local upload first.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

from src.ingestion.chunk import create_chunks_by_title
from src.ingestion.enrich import separate_content_types
from src.ingestion.extract import partition_document
from src.logger import get_logger

logger = get_logger(__name__)

# Committed to git on purpose (unlike model_testing_data/, which holds BYOK
# keys): parsing a PDF loads a torch-based layout model into memory and can
# take 60-90s -- fine locally, but it OOM-kills the deployed dashboard's
# memory-constrained container. Caching a PDF's parsed chunks by content hash
# here lets a demo replay an already-parsed datasheet instantly, without
# re-running the expensive/crash-prone parse on the deployed instance.
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "parsed_cache"


@dataclass
class TestChunk:
    chunk_id: str
    text: str
    tables: list[str] = field(default_factory=list)
    images_b64: list[str] = field(default_factory=list)


def _hash_pdf(pdf_path: Path) -> str:
    """Content hash of a PDF, used as the cache key.

    Hashing content (not filename) means the same datasheet uploaded twice
    under different names still hits the cache, and a changed file under the
    same name correctly misses it.
    """
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]


def _cache_get(pdf_hash: str) -> list[TestChunk] | None:
    path = CACHE_DIR / f"{pdf_hash}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except Exception as e:
        logger.warning("Parse cache file %s is unreadable (%s) -- treating as a miss", path, e)
        return None
    return [TestChunk(**c) for c in raw["chunks"]]


def _cache_put(pdf_hash: str, source_label: str, chunks: list[TestChunk]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{pdf_hash}.json"
    path.write_text(json.dumps(
        {"source_label": source_label, "chunks": [asdict(c) for c in chunks]},
        indent=2,
    ))
    logger.info("Cached %d parsed chunk(s) for %s at %s", len(chunks), source_label, path)


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

    Checks the on-disk parse cache (keyed by PDF content hash) first. On a
    hit, skips the live parse entirely -- this is what makes the deployed
    dashboard's demo path safe (see ``CACHE_DIR``'s docstring above). On a
    miss, parses live and writes the result to the cache for next time.
    ``max_chunks`` is applied after the cache lookup either way, so the
    cache always holds the full, untruncated parse.
    """
    pdf_path = Path(pdf_path)
    pdf_hash = _hash_pdf(pdf_path)

    cached = _cache_get(pdf_hash)
    if cached is not None:
        logger.info("Parse cache hit for %s (hash=%s) -- skipping live parse", pdf_path.name, pdf_hash)
        out = cached
    else:
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

        out = []
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
        _cache_put(pdf_hash, pdf_path.name, out)

    return out[:max_chunks] if max_chunks is not None else out
