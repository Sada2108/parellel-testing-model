"""Colab-oriented ingestion for pre-exported JSON chunk files.

Reads ``chunks_export.json``-style files from a directory and ingests them
into ChromaDB.  Unlike the main pipeline this variant saves images as
separate files (referenced by path in metadata) to avoid bloating the vector
store, and flattens the original content onto the metadata dict directly.

Paths default to Colab-friendly relatives (``../json``, ``../dbv2/...``).

Usage:
    python -m scripts.ingest_colab          # runs with default paths
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from config.settings import CHROMA_COLLECTION_METADATA, EMBEDDING_MODEL
from src.logger import get_logger

logger = get_logger(__name__)


def ingest_json_directory(
    json_directory: str = "../json",
    persist_directory: str = "../dbv2/chroma_db",
    images_directory: str = "../dbv2/images",
) -> Chroma:
    """Read JSON chunk files from a directory and ingest them into ChromaDB.

    Images are decoded from base64 and saved as separate files on disk; the
    metadata stores their paths plus the raw text and tables.

    Args:
        json_directory: Folder containing the exported JSON files.
        persist_directory: Where to persist the Chroma store.
        images_directory: Where to write extracted images.

    Returns:
        The populated Chroma vector store.

    Raises:
        FileNotFoundError: If no JSON files are found in ``json_directory``.
    """
    logger.info("=" * 50)
    logger.info("INGESTION PIPELINE START")
    logger.info("=" * 50)

    logger.info("[1/4] Loading embedding model...")
    embedding_model = HuggingFaceEndpointEmbeddings(model=EMBEDDING_MODEL)
    logger.info("  Model: %s", embedding_model.model)

    logger.info("[2/4] Preparing documents and images...")
    images_dir = Path(images_directory)
    images_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(Path(json_directory).glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_directory}")

    logger.info("  Found %d JSON files", len(json_files))

    documents: list[Document] = []
    total_images = 0
    total_tables = 0
    total_raw_chars = 0

    for json_file in json_files:
        logger.info("  Processing: %s", json_file.name)

        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        logger.info("    Chunks in file: %d", len(chunks))

        for chunk in chunks:
            original = chunk["metadata"]["original_content"]
            source_stem = json_file.stem
            chunk_id = chunk["chunk_id"]

            images_b64 = original.get("images_base64", [])
            tables = original.get("tables_html", [])
            raw_text = original.get("raw_text", "")

            image_paths = []
            for idx, b64 in enumerate(images_b64):
                img_filename = f"{source_stem}_c{chunk_id}_img{idx}.jpg"
                img_path = images_dir / img_filename
                img_path.write_bytes(base64.b64decode(b64))
                image_paths.append(str(img_path))

            total_images += len(images_b64)
            total_tables += len(tables)
            total_raw_chars += len(raw_text)

            metadata = {
                "source": json_file.stem,
                "chunk_id": chunk_id,
                "raw_text": raw_text,
                "tables_html": json.dumps(tables, ensure_ascii=False),
                "image_paths": json.dumps(image_paths, ensure_ascii=False),
                "has_table": len(tables) > 0,
                "has_image": len(images_b64) > 0,
            }

            documents.append(
                Document(
                    page_content=chunk["enhanced_content"],
                    metadata=metadata,
                )
            )

    logger.info("  Document summary:")
    logger.info("    Total chunks: %d", len(documents))
    logger.info("    Total images saved: %d", total_images)
    logger.info("    Total tables: %d", total_tables)
    logger.info("    Total raw text chars: %d", total_raw_chars)

    logger.info("[3/4] Generating embeddings (%d chunks)...", len(documents))
    logger.info("  This may take a while depending on API quota...")

    t0 = time.time()
    try:
        db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_metadata=CHROMA_COLLECTION_METADATA,
        )
        logger.info("  Embeddings completed in %.1fs", time.time() - t0)
        logger.info("  Collection size: %d documents", db._collection.count())

    except Exception as e:
        elapsed = time.time() - t0
        logger.error("ERROR after %.1fs: %s: %s", elapsed, type(e).__name__, e)

        if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower():
            logger.error("  -> QUOTA/RATE LIMIT EXCEEDED")
            logger.error("  -> Wait a few minutes and re-run this cell.")
            logger.error(
                "  -> Or switch to a local embedding model (e.g. sentence-transformers)."
            )
        elif "401" in str(e) or "403" in str(e) or "auth" in str(e).lower():
            logger.error("  -> AUTH ERROR - check your HF_TOKEN")
        elif "readonly" in str(e).lower():
            logger.error("  -> DATABASE LOCKED - delete the persist_directory and re-run.")
        else:
            logger.error("  -> Unexpected error, check the traceback above.")

        raise

    logger.info("[4/4] Verifying...")
    count = db._collection.count()
    logger.info("  Documents in ChromaDB: %d", count)
    if count != len(documents):
        logger.warning("  WARNING: Expected %d but got %d", len(documents), count)

    logger.info("  Images saved to: %s", images_dir.resolve())
    logger.info("  DB saved to: %s", Path(persist_directory).resolve())

    logger.info("=" * 50)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 50)

    return db


if __name__ == "__main__":
    ingest_json_directory()
