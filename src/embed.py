"""Embedding model factory.

Provides a single function to build the embedding model used by both the
ingestion pipeline (when creating the vector store) and the retrieval path
(when loading it back), guaranteeing both sides use the same configuration.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEndpointEmbeddings

from config.settings import EMBEDDING_MODEL
from src.logger import get_logger

logger = get_logger(__name__)


def load_embedding_model() -> HuggingFaceEndpointEmbeddings:
    """Create the HuggingFace embedding model configured in settings.

    Returns:
        A configured :class:`HuggingFaceEndpointEmbeddings` instance ready to
        embed text or load a vector store.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return HuggingFaceEndpointEmbeddings(model=EMBEDDING_MODEL)
