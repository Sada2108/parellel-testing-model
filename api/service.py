"""Backing service for the FastAPI endpoints.

Wraps the existing retrieval stack (``src.retrieval``) behind a lazy-loaded,
thread-safe service so endpoints stay thin.  All heavy work (embedding model
load, Chroma load, LLM calls) runs in a worker thread via ``asyncio.to_thread``
so FastAPI's event loop is never blocked.

Every query reuses the single-trace path from the codebase
(``answer_query_stream``), so the LangSmith UI shows one ``AnswerQuery`` run
with ``RetrieveChunks`` and ``GenerateAnswerStream`` nested underneath.
"""

from __future__ import annotations

import asyncio

from config.settings import RETRIEVAL_FETCH_K, RETRIEVAL_K, RETRIEVAL_SEARCH_TYPE
from src.documents import extract_original_data
from src.embed import load_embedding_model
from src.logger import get_logger
from src.retrieval.generate import answer_query_stream
from src.retrieval.search import build_retriever, load_vector_store, retrieve_chunks

from api.images import ImageRegistry
from api.schemas import ChunkPayload, RetrieveResponse

logger = get_logger(__name__)

# Metadata keys that are excluded from the API payload's ``metadata`` dict to
# keep responses small (the full content is exposed via dedicated fields).
_EXCLUDED_META_KEYS = {"original_content", "raw_text", "tables_html", "images_base64", "image_paths"}


class RAGService:
    """Lazily loads the vector store and exposes retrieve/answer operations."""

    def __init__(self):
        self._db = None
        self._retriever = None
        self._load_lock = asyncio.Lock()
        self.images = ImageRegistry()

    # -- lifecycle ----------------------------------------------------------

    async def ensure_ready(self) -> None:
        """Load the vector store on first use (idempotent, thread-safe)."""
        if self._db is None:
            async with self._load_lock:
                if self._db is None:
                    await asyncio.to_thread(self._load)

    def _load(self) -> None:
        logger.info("Loading embedding model and vector store...")
        model = load_embedding_model()
        self._db = load_vector_store(model)
        self._retriever = build_retriever(self._db)
        logger.info("Vector store ready.")

    def get_retriever(self, top_k: int | None = None):
        """Return a retriever, honouring a per-request ``top_k`` override."""
        if self._db is None:
            raise RuntimeError("Vector store is not loaded")
        if top_k and top_k != RETRIEVAL_K:
            return self._db.as_retriever(
                search_type=RETRIEVAL_SEARCH_TYPE,
                search_kwargs={"k": top_k, "fetch_k": max(top_k, RETRIEVAL_FETCH_K)},
            )
        return self._retriever

    # -- operations ---------------------------------------------------------

    async def retrieve(self, query: str, top_k: int) -> RetrieveResponse:
        """Retrieve chunks for a query without generating an answer."""
        await self.ensure_ready()
        retriever = self.get_retriever(top_k)
        chunks = await asyncio.to_thread(retrieve_chunks, retriever, query)
        return self.to_retrieve_response(query, chunks)

    async def answer(
        self, query: str, top_k: int, summarize_context: bool = False
    ) -> tuple[str, RetrieveResponse, dict]:
        """Retrieve chunks and generate a full answer.

        Uses ``answer_query_stream`` so retrieval and generation stay under a
        single ``AnswerQuery`` trace; the answer is built by consuming every
        streamed token.

        Args:
            summarize_context: See ``generate.generate_answer``. Default False.

        Returns:
            A tuple of ``(answer, retrieve_response, prompt_token_breakdown)``.
        """
        await self.ensure_ready()
        retriever = self.get_retriever(top_k)

        def _run() -> tuple[str, list, dict]:
            stream = answer_query_stream(retriever, query, summarize_context=summarize_context)
            chunks = stream.chunks
            answer_text = "".join(stream)
            return answer_text, chunks, stream.token_breakdown

        answer, chunks, token_breakdown = await asyncio.to_thread(_run)
        return answer, self.to_retrieve_response(query, chunks), token_breakdown

    # -- payload assembly ---------------------------------------------------

    def to_retrieve_response(self, query: str, chunks: list) -> RetrieveResponse:
        """Build the API response for a set of retrieved chunks.

        Registers all chunk images in the :class:`ImageRegistry` and produces
        the ``image_urls`` that reference them.
        """
        images_by_chunk = [extract_original_data(c)["images_base64"] for c in chunks]

        starts: list[int] = []
        offset = 0
        for images in images_by_chunk:
            starts.append(offset)
            offset += len(images)

        response_id = self.images.register(images_by_chunk)

        payloads: list[ChunkPayload] = []
        total_images = 0
        total_tables = 0
        for i, chunk in enumerate(chunks):
            raw = extract_original_data(chunk)
            images = raw["images_base64"]
            tables = raw["tables_html"]
            metadata = {
                k: v
                for k, v in (chunk.metadata or {}).items()
                if k not in _EXCLUDED_META_KEYS
            }
            payloads.append(
                ChunkPayload(
                    chunk_id=i + 1,
                    enhanced_content=chunk.page_content or "",
                    raw_text=raw["raw_text"],
                    tables_html=tables,
                    image_count=len(images),
                    images_base64=images,
                    image_urls=[
                        f"/images/{response_id}/{starts[i] + j}" for j in range(len(images))
                    ],
                    has_table=bool(tables),
                    has_image=bool(images),
                    source=metadata.get("source"),
                    metadata=metadata,
                )
            )
            total_images += len(images)
            total_tables += len(tables)

        return RetrieveResponse(
            query=query,
            num_chunks=len(chunks),
            total_images=total_images,
            total_tables=total_tables,
            response_id=response_id,
            chunks=payloads,
        )
