"""Retrieval pipeline entry point.

Loads the vector store, builds the retriever, and answers a query with the
multimodal LLM.

Usage:
    python -m scripts.query
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embed import load_embedding_model
from src.logger import get_logger
from src.retrieval.generate import answer_query
from src.retrieval.search import build_retriever, load_vector_store

logger = get_logger(__name__)


def main() -> None:
    """Run the interactive query loop: load store -> retrieve -> generate."""
    logger.info("Loading embedding model...")
    embedding_model = load_embedding_model()

    logger.info("Loading vector store...")
    db = load_vector_store(embedding_model)

    retriever = build_retriever(db)
    logger.info("Ready.")

    query = input("Enter your query: ")
    if not query.strip():
        logger.warning("No query provided.")
        return

    final_answer = answer_query(retriever, query)
    print("\n" + "-" * 5)
    print(final_answer)


if __name__ == "__main__":
    main()
