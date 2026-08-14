"""Retrieval package.

Exposes vector store loading, retrieval and answer generation so the UI and
CLI can build a full RAG query in a few calls.
"""

from src.retrieval.generate import (
    answer_query,
    answer_query_stream,
    generate_answer,
    generate_answer_stream,
)
from src.retrieval.search import (
    build_retriever,
    load_vector_store,
    retrieve_chunks,
)

__all__ = [
    "load_vector_store",
    "build_retriever",
    "retrieve_chunks",
    "generate_answer",
    "generate_answer_stream",
    "answer_query",
    "answer_query_stream",
]
