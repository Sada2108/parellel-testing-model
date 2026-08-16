"""RAG-answer prompt building for the Query & Retrieve test call site.

Mirrors the shape of ``src/retrieval/generate.py``'s
``_build_text_prompt`` / ``_collect_images`` / ``_build_message_content``,
but operates on the flat chunk dicts returned by the API's ``/retrieve``
endpoint (``app.api_client.retrieve()``) instead of LangChain ``Document``
objects, so this stays decoupled from the production retrieval module --
nothing in ``src/retrieval`` is imported or touched here.
"""

from __future__ import annotations


def _build_rag_text_prompt(question: str, chunks: list[dict]) -> str:
    parts = [
        "You are given retrieved chunks from a technical document.\n"
        "Each chunk contains:\n"
        "- a searchable summary of the original content,\n"
        "- the original raw text it was extracted from,\n"
        "- optional HTML tables,\n"
        "- optional attached figures/images.\n\n"
        "Treat the summary and raw text together as authoritative representations of "
        "the document -- the raw text is the ground truth the summary was generated from, "
        "so use it to verify or find specifics the summary may have condensed away.\n"
        "Do NOT claim information is absent unless you have examined ALL provided chunks.\n"
        "If a chunk explicitly contains the answer, answer directly using that chunk.\n"
        "Use the text summaries, raw text, and attached images together when answering.\n",
        f"QUESTION: {question}\n",
        "RETRIEVED CONTEXT:",
        "",
    ]

    for i, chunk in enumerate(chunks[:5]):
        enhanced = chunk.get("enhanced_content") or ""
        if enhanced:
            parts.append(f"--- Chunk {i + 1} ---")
            parts.append(f"SUMMARY:\n{enhanced.strip()}\n")

        raw_text = chunk.get("raw_text") or ""
        if raw_text:
            parts.append(f"RAW TEXT:\n{raw_text.strip()}\n")

        tables_html = chunk.get("tables_html") or []
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


def _collect_rag_images(chunks: list[dict]) -> list[dict]:
    image_blocks: list[dict] = []
    for chunk in chunks:
        for b64 in chunk.get("images_base64") or []:
            image_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
    return image_blocks


def build_rag_test_messages(
    question: str, chunks: list[dict], vision: bool
) -> list[dict]:
    """Build the chat ``messages`` list for testing a model at the
    Query & Retrieve call site.

    Args:
        question: The user's question.
        chunks: Retrieved chunk payloads, as returned by the API's
            ``/retrieve`` endpoint (or entered manually in the same shape).
        vision: Whether to attach chunk images. Non-vision models get a
            text-only prompt, same convention as the summary call site's
            ``_build_messages``.

    Returns:
        A ``messages`` list ready to pass to ``harness.call_one_model``.
    """
    prompt_text = _build_rag_text_prompt(question, chunks)

    if not vision:
        return [{"role": "user", "content": prompt_text}]

    content: list[dict] = [{"type": "text", "text": prompt_text}]
    content.extend(_collect_rag_images(chunks))
    return [{"role": "user", "content": content}]
