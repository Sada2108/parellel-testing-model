"""Answer generation using a multimodal LLM.

Builds the prompt from retrieved chunks (enhanced summaries + HTML tables +
base64 images), calls Gemini to generate an answer, and wraps both retrieval
and generation under a single LangSmith trace so the whole query shows up as
one run in the UI.
"""

from __future__ import annotations

import base64

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from langsmith.run_helpers import set_tracing_parent

from config.settings import GENERATION_MODEL, GENERATION_TEMPERATURE
from src.documents import extract_original_data
from src.logger import get_logger
from src.retrieval.search import retrieve_chunks
from src.tracing import summarize_chunks

logger = get_logger(__name__)


def _compress_chunk_summary(text: str, llm: ChatGoogleGenerativeAI, target_words: int = 100) -> str:
    """Compress one retrieved chunk's summary for this query's context window.

    Query-time only: this shortens what goes into *this one prompt*, not the
    stored chunk (the vector store / chunks_huggingface.json keep the full
    original summary untouched). Distinct from Phase 3A's ingestion-time
    length limiting, which reshapes what gets stored forever. Uses a soft
    word-count instruction, same non-truncating philosophy as 3A -- not a
    hard token cutoff, and no change to the chunk-level summary itself.

    Falls back to the original text if compression fails for any reason --
    a failed compression call should degrade to "use full context", not
    drop the chunk's content from the answer entirely.
    """
    if not text:
        return text
    try:
        prompt = (
            f"Compress the following technical summary to at most {target_words} words. "
            "Keep only facts, numbers, and specifics relevant to being retrieved by a "
            "search query -- drop restated boilerplate and filler.\n\n"
            f"{text}"
        )
        response = llm.invoke([HumanMessage(content=[{"type": "text", "text": prompt}])])
        compressed = _extract_text(response.content).strip()
        return compressed or text
    except Exception as e:
        logger.warning("Context compression failed for a chunk, using full text: %s", e)
        return text


def _prompt_token_breakdown(chunks: list[Document], query: str, chunk_text_chars: int) -> dict:
    """Rough char-length-based estimate of prompt size by component.

    Not a real tokenizer count -- this codebase has no local tokenizer, and
    a real per-component count isn't available from a single completed API
    call (usage is only reported as one total). Char/4 is a standard rough
    English-text token approximation; base64 image data uses the same
    divisor purely as a size proxy (images are priced very differently by
    the provider, not per this estimate) so the *relative* share it takes
    up is still visible -- the diagnostic's actual point per the build doc.

    ``chunk_text_chars`` is the actual chunk-text length used in the built
    prompt (passed in rather than recomputed from ``chunks`` here) so this
    reflects post-compression size when ``summarize_context`` was applied --
    re-reading ``chunk.page_content`` directly would silently ignore the
    compression and always report the original, pre-compression size.
    """
    instruction_chars = 400  # roughly the fixed instruction/question wrapper text
    table_chars = 0
    image_chars = 0

    for chunk in chunks[:5]:
        original = extract_original_data(chunk)
        table_chars += sum(len(t) for t in original["tables_html"])
        image_chars += sum(len(b64) for b64 in original["images_base64"])

    instruction_chars += len(query)
    total_chars = instruction_chars + chunk_text_chars + table_chars + image_chars

    def _pct(n: int) -> float:
        return round(100 * n / total_chars, 1) if total_chars else 0.0

    return {
        "instruction_pct": _pct(instruction_chars),
        "chunk_text_pct": _pct(chunk_text_chars),
        "tables_pct": _pct(table_chars),
        "images_pct": _pct(image_chars),
        "est_total_tokens": round(total_chars / 4),
    }


def _build_text_prompt(
    chunks: list[Document],
    query: str,
    summarize_context: bool = False,
    llm: ChatGoogleGenerativeAI | None = None,
    stats: dict | None = None,
) -> str:
    """Build the text portion of the prompt from retrieved chunks.

    Sends enhanced summaries as primary context, with HTML tables appended.
    Images are sent separately via :func:`_collect_images`.

    Note: only the first 5 chunks are included to keep the prompt size
    predictable for the model.

    ``summarize_context=False`` (default) is today's behavior, unchanged.
    When ``True``, each chunk's enhanced summary is compressed for this
    query's prompt only (requires ``llm`` -- reuses the same Gemini client
    already created for the actual answer, no extra client setup).

    ``stats``, if given, gets ``stats["chunk_text_chars"]`` set to the
    actual (post-compression, when applicable) chunk-text length used --
    lets :func:`_prompt_token_breakdown` reflect real compression, since it
    can't re-derive that from ``chunks`` alone (that still holds the
    original, uncompressed summaries).
    """
    parts = [
        "You are given retrieved chunks from a technical document.\n"
        "Each chunk contains:\n"
        "- a searchable summary of the original content,\n"
        "- optional HTML tables,\n"
        "- optional attached figures/images.\n\n"
        "Treat the summaries as authoritative representations of the document.\n"
        "Do NOT claim information is absent unless you have examined ALL provided chunks.\n"
        "If a chunk explicitly contains the answer, answer directly using that chunk.\n"
        "Use both the text summaries and the attached images when answering.\n",
        f"QUESTION: {query}\n",
        "RETRIEVED CONTEXT:",
        "",
    ]

    chunk_text_chars = 0
    for i, chunk in enumerate(chunks[:5]):
        enhanced = chunk.page_content
        if enhanced:
            if summarize_context and llm is not None:
                enhanced = _compress_chunk_summary(enhanced, llm)
            chunk_text_chars += len(enhanced)
            parts.append(f"--- Chunk {i + 1} ---")
            parts.append(f"SUMMARY:\n{enhanced.strip()}\n")

        tables_html = extract_original_data(chunk)["tables_html"]
        if tables_html:
            parts.append("TABLES:")
            for j, table in enumerate(tables_html):
                parts.append(f"Table {j + 1}:\n{table}\n")

        parts.append("")

    if stats is not None:
        stats["chunk_text_chars"] = chunk_text_chars

    parts.append(
        "The images above are figures from the document. "
        "Use the summaries, tables, and images together to provide a detailed answer."
    )

    return "\n".join(parts)


def _collect_images(chunks: list[Document]) -> list[dict]:
    """Collect all base64 images from chunks into message content blocks."""
    image_blocks: list[dict] = []
    for chunk in chunks:
        for b64 in extract_original_data(chunk)["images_base64"]:
            image_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
    return image_blocks


def _build_message_content(
    chunks: list[Document],
    query: str,
    summarize_context: bool = False,
    llm: ChatGoogleGenerativeAI | None = None,
    stats: dict | None = None,
) -> list[dict]:
    """Build the full multimodal message content list (text + images)."""
    content: list[dict] = [
        {"type": "text", "text": _build_text_prompt(chunks, query, summarize_context, llm, stats)}
    ]
    content.extend(_collect_images(chunks))
    return content


def _create_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini LLM instance used for generation."""
    logger.info("  Using model: %s", GENERATION_MODEL)
    return ChatGoogleGenerativeAI(
        model=GENERATION_MODEL,
        temperature=GENERATION_TEMPERATURE,
    )


def _extract_text(content: str | list) -> str:
    """Normalize a LangChain message ``.content`` value into plain text.

    Older Gemini models (and most other chat models) return ``.content`` as
    a plain string. Newer Gemini models return a list of content blocks
    instead (e.g. ``[{"type": "text", "text": "..."}]``), sometimes
    interleaved with non-text blocks (thought signatures) or empty-text
    blocks. Handle both shapes so this keeps working regardless of which
    shape the SDK/model gives back.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "".join(parts)
    return ""


def _save_prompt_debug(message_content: list[dict]) -> None:
    """Dump the full text prompt to ``last_prompt.txt`` for inspection."""
    from pathlib import Path

    text_content = next(x["text"] for x in message_content if x["type"] == "text")
    image_count = sum(1 for x in message_content if x["type"] == "image_url")
    Path("last_prompt.txt").write_text(text_content, encoding="utf-8")
    logger.info(
        "Prompt saved to last_prompt.txt (%d chars, %d images)",
        len(text_content),
        image_count,
    )


@traceable(run_type="llm", name="GenerateAnswer")
def generate_answer(
    chunks: list[Document], query: str, verbose: bool = False, summarize_context: bool = False
) -> str:
    """Generate a final answer using the multimodal LLM.

    Args:
        chunks: Retrieved document chunks.
        query: The original user query.
        verbose: If True, logs additional debug detail.
        summarize_context: If True, compress each chunk's summary for this
            query's prompt only (see :func:`_compress_chunk_summary`).
            Default False -- unchanged behavior.

    Returns:
        The generated answer string.
    """
    try:
        chunk_summaries = summarize_chunks(chunks)
        total_images = sum(c.get("image_count", 0) for c in chunk_summaries)
        total_tables = sum(c.get("table_count", 0) for c in chunk_summaries)

        llm = _create_llm()
        stats: dict = {}
        message_content = _build_message_content(chunks, query, summarize_context, llm, stats)
        _save_prompt_debug(message_content)
        logger.info(
            "Prompt token breakdown (est.): %s",
            _prompt_token_breakdown(chunks, query, stats.get("chunk_text_chars", 0)),
        )

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        logger.info(
            "LangSmith trace: %d chunks, %d images, %d tables",
            len(chunks),
            total_images,
            total_tables,
        )
        return _extract_text(response.content)

    except Exception as e:
        error_msg = f"Answer generation failed: {e}"
        if verbose:
            logger.exception(error_msg)
        else:
            logger.warning(error_msg)
        return f"Sorry, could not complete response due to: {e}"


def _collect_image_attachments(chunks: list[Document]) -> dict:
    """Gather images from chunks as binary attachments for LangSmith traces.

    Uses :func:`extract_original_data` to resolve either storage schema
    (file-path or inline base64) to image bytes.
    """
    attachments: dict = {}
    img_idx = 0
    for chunk in chunks:
        for b64_str in extract_original_data(chunk)["images_base64"]:
            try:
                attachments[f"chunk_img_{img_idx}"] = ("image/jpeg", base64.b64decode(b64_str))
                img_idx += 1
            except Exception:
                logger.warning("Could not decode image attachment for trace")
    return attachments


def _attach_images_to_trace(run_tree, chunks: list[Document]) -> None:
    """Attach retrieved chunk images to ``run_tree`` if any are available."""
    if run_tree is None:
        return
    attachments = _collect_image_attachments(chunks)
    if attachments:
        run_tree.attachments = attachments


class _StreamWrapper:
    """Iterator adapter that keeps the LangSmith parent context alive.

    ``generate_answer_stream`` is a generator, so LangSmith defers creating
    its trace run until the first ``next()`` call.  By the time that happens
    the original parent context (inside ``answer_query_stream``) is gone, so
    we re-activate it around every ``next()`` to keep retrieval + generation
    nested under the same ``AnswerQuery`` root run.
    """

    def __init__(
        self,
        gen,
        chunks: list[Document],
        parent=None,
        query: str = "",
        stats: dict | None = None,
    ):
        self._gen = gen
        self.chunks = chunks
        self._query = query
        self._stats = stats if stats is not None else {}
        self._parent = parent

    @property
    def token_breakdown(self) -> dict:
        """Computed lazily from ``_stats``, which ``generate_answer_stream``
        populates with the real (post-compression) chunk-text size as it
        runs -- accessing this before the stream is fully consumed will see
        ``chunk_text_chars=0`` since generation hasn't happened yet."""
        return _prompt_token_breakdown(self.chunks, self._query, self._stats.get("chunk_text_chars", 0))

    def __iter__(self):
        return self

    def __next__(self):
        if self._parent is not None:
            with set_tracing_parent(self._parent):
                return next(self._gen)
        return next(self._gen)


@traceable(run_type="chain", name="AnswerQuery", dangerously_allow_filesystem=True)
def answer_query(retriever, query: str, run_tree=None, summarize_context: bool = False) -> str:
    """Retrieve chunks then generate an answer under a single LangSmith trace.

    Images from retrieved chunks are attached to the trace so they render in
    the LangSmith UI.  ``run_tree`` is injected by LangSmith.

    Args:
        retriever: Vector store retriever.
        query: The user's question.
        summarize_context: See :func:`generate_answer`. Default False.

    Returns:
        The generated answer string.
    """
    chunks = retrieve_chunks(retriever, query)
    _attach_images_to_trace(run_tree, chunks)
    return generate_answer(chunks, query, summarize_context=summarize_context)


@traceable(run_type="chain", name="AnswerQuery", dangerously_allow_filesystem=True)
def answer_query_stream(retriever, query: str, run_tree=None, summarize_context: bool = False):
    """Retrieve chunks then stream an answer under a single LangSmith trace.

    Yields answer tokens as they are produced.  The retrieved chunks are
    available via the returned generator's ``chunks`` attribute once
    retrieval completes.  ``run_tree`` is injected by LangSmith.

    Args:
        retriever: Vector store retriever.
        query: The user's question.
        summarize_context: See :func:`generate_answer`. Default False.

    Yields:
        Answer token strings.
    """
    chunks = retrieve_chunks(retriever, query)
    _attach_images_to_trace(run_tree, chunks)

    stats: dict = {}
    return _StreamWrapper(
        generate_answer_stream(chunks, query, summarize_context=summarize_context, stats=stats),
        chunks,
        run_tree,
        query=query,
        stats=stats,
    )


@traceable(run_type="llm", name="GenerateAnswerStream")
def generate_answer_stream(
    chunks: list[Document],
    query: str,
    verbose: bool = False,
    summarize_context: bool = False,
    stats: dict | None = None,
):
    """Generate a streaming answer using the multimodal LLM.

    Yields answer tokens as they are generated by the LLM.

    Args:
        chunks: Retrieved document chunks.
        query: The original user query.
        verbose: If True, logs additional debug detail.
        summarize_context: See :func:`generate_answer`. Default False.
        stats: See :func:`_build_text_prompt` -- shared dict this writes
            ``chunk_text_chars`` into, so a caller holding the same dict
            (e.g. ``_StreamWrapper``) can read the real post-compression
            size once generation has started.

    Yields:
        Answer token strings.
    """
    try:
        chunk_summaries = summarize_chunks(chunks)
        total_images = sum(c.get("image_count", 0) for c in chunk_summaries)
        total_tables = sum(c.get("table_count", 0) for c in chunk_summaries)

        llm = _create_llm()

        logger.info("  %d chunks received by generation", len(chunks))
        for ci, c in enumerate(chunks[:5]):
            od = extract_original_data(c)
            logger.debug(
                "    Chunk %d: summary=%d chars, tables=%d, images=%d",
                ci + 1,
                len(c.page_content or ""),
                len(od["tables_html"]),
                len(od["images_base64"]),
            )

        message_content = _build_message_content(chunks, query, summarize_context, llm, stats)
        _save_prompt_debug(message_content)
        logger.info(
            "Prompt token breakdown (est.): %s",
            _prompt_token_breakdown(chunks, query, (stats or {}).get("chunk_text_chars", 0)),
        )

        logger.info(
            "LangSmith trace: %d chunks, %d images, %d tables",
            len(chunks),
            total_images,
            total_tables,
        )

        message = HumanMessage(content=message_content)
        for chunk in llm.stream([message]):
            text = _extract_text(chunk.content)
            if text:
                yield text

    except Exception as e:
        logger.warning("Answer generation failed: %s", e)
        yield f"Sorry, could not complete response due to: {e}"
