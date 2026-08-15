"""Content-type separation and AI-enhanced summary generation.

The second half of ingestion: each unstructured chunk is analysed for its
content types (text, tables, images), and a multimodal vision model generates
a searchable summary that captures both the textual data and the visual
content.  The result is wrapped into a LangChain ``Document`` whose
``metadata["original_content"]`` stores the raw payload for later retrieval
and UI rendering.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.documents import Document
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

from config.settings import (
    ENHANCEMENT_BASE_URL,
    ENHANCEMENT_MAX_TOKENS,
    ENHANCEMENT_MODEL,
    ENHANCEMENT_TEMPERATURE,
    HF_TOKEN,
)
from src.logger import get_logger
from src.text_utils import strip_think_blocks

logger = get_logger(__name__)


def separate_content_types(chunk: Any) -> dict:
    """Analyse which content types are present in a chunk.

    Inspects the ``orig_elements`` attached by Unstructured to split a chunk
    into its textual content, HTML tables and base64-encoded images.

    Args:
        chunk: An Unstructured chunk with a ``text`` and ``metadata``.

    Returns:
        Dict with keys ``text`` (str), ``tables`` (list[str]),
        ``images`` (list[str]) and ``types`` (set of type names).
    """
    content_data: dict = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"],
    }

    orig_elements = getattr(chunk.metadata, "orig_elements", None)
    if orig_elements is None:
        return content_data

    for element in orig_elements:
        element_type = type(element).__name__

        if element_type == "Table":
            content_data["types"].append("table")
            table_html = getattr(element.metadata, "text_as_html", element.text)
            content_data["tables"].append(table_html)

        elif element_type == "Image":
            img_b64 = getattr(element.metadata, "image_base64", None)
            if img_b64:
                content_data["types"].append("image")
                content_data["images"].append(img_b64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data


def _create_enhancement_client() -> OpenAI:
    """Create an OpenAI-compatible client for the enhancement model.

    The raw client is wrapped with ``wrap_openai`` so that LangSmith renders
    the base64 images inside the multimodal trace.
    """
    raw = OpenAI(api_key=HF_TOKEN, base_url=ENHANCEMENT_BASE_URL)
    return wrap_openai(raw, chat_name="EnhancementVisionLLM")


def _build_enhancement_prompt(
    text: str, tables: list[str], max_summary_words: int | None = None
) -> str:
    """Build the text prompt for AI-based content enhancement.

    ``max_summary_words`` is ``None`` by default -- unbounded, today's
    behavior, unchanged. When set, switches to a length-targeted prompt:
    a soft word-count instruction plus a template that scales down for
    simple content instead of always emitting maximal structure (the
    failure mode this was built to fix: a short chunk like a package
    outline getting a full "DOCUMENT IDENTIFICATION & METADATA" section
    with nothing to actually put in it). Deliberately not a hard
    ``max_tokens`` cutoff -- that truncates mid-sentence instead of
    producing a complete, shorter summary.
    """
    prompt = (
        "You are creating a searchable description for document content retrieval.\n\n"
        "CONTENT TO ANALYZE:\n"
        "TEXT CONTENT:\n"
        f"{text}\n\n"
    )

    if tables:
        prompt += "TABLES:\n"
        for i, table in enumerate(tables):
            prompt += f"Table {i + 1}:\n{table}\n\n"

    if max_summary_words is None:
        prompt += (
            "YOUR TASK:\n"
            "Generate a comprehensive, searchable description that covers:\n\n"
            "1. Key facts, numbers, and data points from text and tables\n"
            "2. Main topics and concepts discussed\n"
            "3. Questions this content could answer\n"
            "4. Visual content analysis (charts, diagrams, patterns in images)\n"
            "5. Alternative search terms users might use\n\n"
            "Make it detailed and searchable - prioritize findability over brevity.\n\n"
            "SEARCHABLE DESCRIPTION:"
        )
    else:
        prompt += (
            "YOUR TASK:\n"
            f"Write a searchable description in no more than {max_summary_words} words. "
            "Include only information actually present in the content above; omit "
            "boilerplate section headers and filler language not warranted by the "
            "content's own density -- a short, simple chunk should get a short "
            "summary, not padded-out sections.\n\n"
            "Use only the fields below that are actually relevant -- skip any that "
            "don't apply rather than writing \"N/A\" or leaving them empty:\n"
            "- IDENTIFICATION: part number, document id, title, or subject, if present\n"
            "- KEY SPECS: electrical/physical/numeric values, ratings, or parameters, if present\n"
            "- TOPICS: what this content covers or what questions it could answer\n"
            "- NOTES: anything else notable (visual content, tables, alternative search terms)\n\n"
            "SEARCHABLE DESCRIPTION:"
        )

    return prompt


@traceable(run_type="llm", name="AIEnhancedSummary")
def create_ai_enhanced_summary(
    text: str, tables: list[str], images: list[str], max_summary_words: int | None = None
) -> str:
    """Create an AI-enhanced searchable summary for mixed content.

    Args:
        text: Raw OCR text of the chunk.
        tables: List of HTML table strings.
        images: List of base64-encoded image strings.
        max_summary_words: Soft word-count target (see
            :func:`_build_enhancement_prompt`). ``None`` (default) is
            today's unbounded behavior, unchanged.

    Returns:
        The enhanced summary text, or a fallback summary on failure (or on
        a response with no real content -- see below) so the pipeline can
        continue without the enrichment model.
    """
    fallback = f"{text[:300]}..."
    if tables:
        fallback += f" [Contains {len(tables)} table(s)]"
    if images:
        fallback += f" [Contains {len(images)} image(s)]"

    try:
        client = _create_enhancement_client()
        prompt_text = _build_enhancement_prompt(text, tables, max_summary_words)

        content: list[dict] = [{"type": "text", "text": prompt_text}]
        for b64 in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        response = client.chat.completions.create(
            model=ENHANCEMENT_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=ENHANCEMENT_TEMPERATURE,
            max_tokens=ENHANCEMENT_MAX_TOKENS,
        )
        # Strip any <think>...</think> reasoning block before it's stored.
        # A no-op for models/providers that don't emit one (verified
        # empirically against a real call to the current ENHANCEMENT_MODEL,
        # GLM-4.5V via the HF router -- plain text, no <think> tags) --
        # this is defensive against a future model swap, not a fix for a
        # currently-reproducing bug in this specific model.
        cleaned = strip_think_blocks(response.choices[0].message.content or "")
        if not cleaned:
            logger.warning(
                "AI summary produced no content outside its reasoning block -- using fallback"
            )
            return fallback
        return cleaned

    except Exception as e:
        logger.warning("AI summary failed for chunk: %s", e)
        return fallback


@traceable(run_type="chain", name="SummariseChunks")
def summarise_chunks(chunks: list, max_summary_words: int | None = None) -> list[Document]:
    """Process all chunks with AI summaries and wrap them as LangChain Documents.

    Each Document stores the enhanced content as ``page_content`` and the full
    original data (raw text, tables, images) as JSON in ``metadata``.

    Args:
        chunks: List of unstructured chunks.
        max_summary_words: Passed through to :func:`create_ai_enhanced_summary`.
            ``None`` (default) is today's unbounded behavior, unchanged --
            production call sites that don't pass this get identical output.

    Returns:
        List of LangChain ``Document`` objects ready for vector storage.
    """
    logger.info("Processing chunks with AI summaries...")
    langchain_documents: list[Document] = []

    for i, chunk in enumerate(chunks):
        current = i + 1
        total = len(chunks)
        logger.info("  Processing chunk %d/%d", current, total)

        content_data = separate_content_types(chunk)
        logger.debug(
            "    Types: %s  |  Tables: %d  |  Images: %d",
            content_data["types"],
            len(content_data["tables"]),
            len(content_data["images"]),
        )

        if content_data["tables"] or content_data["images"]:
            logger.debug("    -> Creating AI summary for mixed content...")
            enhanced = create_ai_enhanced_summary(
                content_data["text"],
                content_data["tables"],
                content_data["images"],
                max_summary_words,
            )
            logger.debug("    -> Enhanced: %s...", enhanced[:150])
        else:
            logger.debug("    -> Using raw text (no tables/images)")
            enhanced = content_data["text"]

        doc = Document(
            page_content=enhanced,
            metadata={
                "original_content": json.dumps(
                    {
                        "raw_text": content_data["text"],
                        "tables_html": content_data["tables"],
                        "images_base64": content_data["images"],
                    }
                )
            },
        )
        langchain_documents.append(doc)

    logger.info("Processed %d chunks", len(langchain_documents))
    return langchain_documents
