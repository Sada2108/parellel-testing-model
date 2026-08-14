"""Vector store loading and document retrieval.

Loads the persisted Chroma store and provides an MMR retriever used by the
answer generation layer.  ``retrieve_chunks`` is a traced run so retrieval
shows up as its own child run inside the ``AnswerQuery`` trace.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langsmith import traceable

from config.settings import (
    CHROMA_PERSIST_DIR,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_K,
    RETRIEVAL_SEARCH_TYPE,
)
from src.logger import get_logger
from src.tracing import summarize_chunks

logger = get_logger(__name__)


def load_vector_store(embedding_model: HuggingFaceEndpointEmbeddings) -> Chroma:
    """Load the persisted Chroma vector store.

    Args:
        embedding_model: The embedding model to query with (must match the
            one used during ingestion).

    Returns:
        The loaded Chroma vector store.
    """
    logger.info("Loading vector store from %s", CHROMA_PERSIST_DIR)
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model,
    )


def build_retriever(db: Chroma):
    """Build an MMR retriever from the vector store.

    Args:
        db: The loaded Chroma vector store.

    Returns:
        A retriever configured with the retrieval settings in ``settings.py``.
    """
    logger.debug(
        "Building retriever (search_type=%s, k=%d, fetch_k=%d)",
        RETRIEVAL_SEARCH_TYPE,
        RETRIEVAL_K,
        RETRIEVAL_FETCH_K,
    )
    return db.as_retriever(
        search_type=RETRIEVAL_SEARCH_TYPE,
        search_kwargs={
            "k": RETRIEVAL_K,
            "fetch_k": RETRIEVAL_FETCH_K,
        },
    )


@traceable(run_type="retriever", name="RetrieveChunks")
def retrieve_chunks(retriever, query: str) -> list[Document]:
    """Run retrieval and return matching chunks.

    Args:
        retriever: The configured retriever.
        query: The user's question.

    Returns:
        List of retrieved document chunks.
    """
    chunks = retriever.invoke(query)
    chunk_summaries = summarize_chunks(chunks)
    total_images = sum(c.get("image_count", 0) for c in chunk_summaries)
    total_tables = sum(c.get("table_count", 0) for c in chunk_summaries)
    logger.info(
        "Retrieved %d chunks (%d images, %d tables)",
        len(chunks),
        total_images,
        total_tables,
    )
    return chunks
