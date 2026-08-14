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


def _build_text_prompt(chunks: list[Document], query: str) -> str:
    """Build the text portion of the prompt from retrieved chunks.

    Sends enhanced summaries as primary context, with HTML tables appended.
    Images are sent separately via :func:`_collect_images`.

    Note: only the first 5 chunks are included to keep the prompt size
    predictable for the model.
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

    for i, chunk in enumerate(chunks[:5]):
        enhanced = chunk.page_content
        if enhanced:
            parts.append(f"--- Chunk {i + 1} ---")
            parts.append(f"SUMMARY:\n{enhanced.strip()}\n")

        tables_html = extract_original_data(chunk)["tables_html"]
        if tables_html:
            parts.append("TABLES:")
            for j, table in enumerate(tables_html):
                parts.append(f"Table {j + 1}:\n{table}\n")

        parts.append("")

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


def _build_message_content(chunks: list[Document], query: str) -> list[dict]:
    """Build the full multimodal message content list (text + images)."""
    content: list[dict] = [{"type": "text", "text": _build_text_prompt(chunks, query)}]
    content.extend(_collect_images(chunks))
    return content


def _create_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini LLM instance used for generation."""
    logger.info("  Using model: %s", GENERATION_MODEL)
    return ChatGoogleGenerativeAI(
        model=GENERATION_MODEL,
        temperature=GENERATION_TEMPERATURE,
    )


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
    chunks: list[Document], query: str, verbose: bool = False
) -> str:
    """Generate a final answer using the multimodal LLM.

    Args:
        chunks: Retrieved document chunks.
        query: The original user query.
        verbose: If True, logs additional debug detail.

    Returns:
        The generated answer string.
    """
    try:
        chunk_summaries = summarize_chunks(chunks)
        total_images = sum(c.get("image_count", 0) for c in chunk_summaries)
        total_tables = sum(c.get("table_count", 0) for c in chunk_summaries)

        llm = _create_llm()
        message_content = _build_message_content(chunks, query)
        _save_prompt_debug(message_content)

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        logger.info(
            "LangSmith trace: %d chunks, %d images, %d tables",
            len(chunks),
            total_images,
            total_tables,
        )
        return response.content

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

    def __init__(self, gen, chunks: list[Document], parent=None):
        self._gen = gen
        self.chunks = chunks
        self._parent = parent

    def __iter__(self):
        return self

    def __next__(self):
        if self._parent is not None:
            with set_tracing_parent(self._parent):
                return next(self._gen)
        return next(self._gen)


@traceable(run_type="chain", name="AnswerQuery", dangerously_allow_filesystem=True)
def answer_query(retriever, query: str, run_tree=None) -> str:
    """Retrieve chunks then generate an answer under a single LangSmith trace.

    Images from retrieved chunks are attached to the trace so they render in
    the LangSmith UI.  ``run_tree`` is injected by LangSmith.

    Args:
        retriever: Vector store retriever.
        query: The user's question.

    Returns:
        The generated answer string.
    """
    chunks = retrieve_chunks(retriever, query)
    _attach_images_to_trace(run_tree, chunks)
    return generate_answer(chunks, query)


@traceable(run_type="chain", name="AnswerQuery", dangerously_allow_filesystem=True)
def answer_query_stream(retriever, query: str, run_tree=None):
    """Retrieve chunks then stream an answer under a single LangSmith trace.

    Yields answer tokens as they are produced.  The retrieved chunks are
    available via the returned generator's ``chunks`` attribute once
    retrieval completes.  ``run_tree`` is injected by LangSmith.

    Args:
        retriever: Vector store retriever.
        query: The user's question.

    Yields:
        Answer token strings.
    """
    chunks = retrieve_chunks(retriever, query)
    _attach_images_to_trace(run_tree, chunks)

    return _StreamWrapper(
        generate_answer_stream(chunks, query),
        chunks,
        run_tree,
    )


@traceable(run_type="llm", name="GenerateAnswerStream")
def generate_answer_stream(
    chunks: list[Document], query: str, verbose: bool = False
):
    """Generate a streaming answer using the multimodal LLM.

    Yields answer tokens as they are generated by the LLM.

    Args:
        chunks: Retrieved document chunks.
        query: The original user query.
        verbose: If True, logs additional debug detail.

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

        message_content = _build_message_content(chunks, query)
        _save_prompt_debug(message_content)

        logger.info(
            "LangSmith trace: %d chunks, %d images, %d tables",
            len(chunks),
            total_images,
            total_tables,
        )

        message = HumanMessage(content=message_content)
        for chunk in llm.stream([message]):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        logger.warning("Answer generation failed: %s", e)
        yield f"Sorry, could not complete response due to: {e}"
