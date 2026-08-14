"""Orchestrator: ingestion pipeline and vector store creation.

Ties the extraction, chunking, enrichment and embedding steps together and
offers two entry points: :func:`create_vector_store` to embed an already
processed document list, and :func:`run_complete_ingestion_pipeline` to run
everything from a PDF in one call.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langsmith import traceable

from config.settings import CHROMA_COLLECTION_METADATA, CHROMA_PERSIST_DIR
from src.embed import load_embedding_model
from src.ingestion.chunk import create_chunks_by_title
from src.ingestion.enrich import summarise_chunks
from src.ingestion.export import export_chunks_to_json
from src.ingestion.extract import partition_document
from src.logger import get_logger

logger = get_logger(__name__)


@traceable(run_type="chain", name="CreateVectorStore")
def create_vector_store(
    documents: list[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """Create and persist a ChromaDB vector store from documents.

    Args:
        documents: List of LangChain Documents.
        persist_directory: Directory to persist the vector store.

    Returns:
        The created Chroma vector store.
    """
    logger.info("Creating embeddings and storing in ChromaDB...")

    embedding_model = load_embedding_model()

    logger.info("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata=CHROMA_COLLECTION_METADATA,
    )
    logger.info("--- Finished creating vector store ---")

    logger.info("Vector store saved to %s", persist_directory)
    return vectorstore


@traceable(run_type="chain", name="FullIngestionPipeline")
def run_complete_ingestion_pipeline(pdf_path: str) -> Chroma:
    """Run the full RAG ingestion pipeline end-to-end.

    Steps: partition -> chunk -> AI summarise -> export -> vector store.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        The Chroma vector store ready for retrieval.
    """
    logger.info("Starting RAG Ingestion Pipeline")
    logger.info("=" * 50)

    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    summarised = summarise_chunks(chunks)
    export_chunks_to_json(summarised, filename="chunks_huggingface.json")
    db = create_vector_store(summarised, persist_directory=CHROMA_PERSIST_DIR)

    logger.info("Pipeline completed successfully!")
    return db
