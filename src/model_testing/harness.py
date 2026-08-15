"""Parallel fan-out: the same chunk, run through every registered model.

This is the actual "parallel testing" your senior meant: one input, N
models called concurrently, each one's tokens/cost/latency/output captured
so they land on the same leaderboard. LiteLLM is the layer that makes "any
model, paste a key" possible without per-provider code -- it speaks one
OpenAI-compatible call to every provider and already knows how to parse
token usage out of the response.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import litellm

import config.settings as _settings  # side effect: load_dotenv(), so .env-sourced keys land in os.environ
from src.ingestion.enrich import _build_enhancement_prompt
from src.logger import get_logger

from .datasheet_input import TestChunk

logger = get_logger(__name__)

# LiteLLM is chatty by default about missing prices / unmapped models --
# those aren't errors here, they're expected for brand new or custom models.
litellm.suppress_debug_info = True

# LangSmith tracing via LiteLLM's built-in callback (not the LangSmith SDK's
# own LANGSMITH_TRACING flag/@traceable -- using both at once on the same
# calls produces duplicate traces). Optional: absent key just means the
# callback never fires, not a hard dependency for running tests.
#
# NOTE: once enabled, full prompts and responses leave this machine and are
# sent to LangSmith's cloud for every model call (including judge calls).
# Acceptable here (test harness on public datasheets, not user data) --
# keep this note if templating this pattern elsewhere with real user data.
LANGSMITH_ENABLED = bool(_settings.LANGSMITH_API_KEY)
if LANGSMITH_ENABLED:
    os.environ.setdefault("LANGSMITH_API_KEY", _settings.LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_PROJECT", "parallel-model-test")
    litellm.success_callback = ["langsmith"]
else:
    logger.info("LANGSMITH_API_KEY not set -- LiteLLM->LangSmith tracing disabled for this harness.")


def build_langsmith_metadata(
    session_id: str,
    call_site: str,
    model_label: str,
    provider: str,
    chunk_id: str,
    datasheet_source: str,
    tested_by: str,
) -> dict:
    """Build the ``metadata`` dict LiteLLM's langsmith callback reads.

    Verified against the installed litellm's actual source
    (litellm/integrations/langsmith.py): ``session_id`` groups every call
    in one parallel-run batch into a single LangSmith session/trace tree;
    ``run_name`` sets the per-call trace name; ``tags`` maps to
    ``request_tags`` (read via ``metadata.get("tags", [])`` in
    ``litellm_logging.py``); any other keys land in the trace's ``extra``
    metadata, which is where chunk_id/datasheet_source/tested_by go.
    """
    return {
        "session_id": session_id,
        "run_name": f"{call_site}:{model_label}",
        "tags": [model_label, provider, call_site],
        "chunk_id": chunk_id,
        "datasheet_source": datasheet_source,
        "tested_by": tested_by,
    }


@dataclass
class ModelResult:
    model_row_id: int
    model_label: str
    provider: str
    model_id: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[float] = None
    output_text: Optional[str] = None
    error: Optional[str] = None


def _litellm_model_string(provider: str, model_id: str) -> str:
    if provider in ("groq", "gemini", "openai", "anthropic", "huggingface"):
        return f"{provider}/{model_id}"
    # "custom" (or anything else): route through the generic OpenAI-compatible
    # path and rely on api_base to point at the right endpoint.
    return f"openai/{model_id}"


def _build_messages(
    chunk: TestChunk, vision: bool, max_summary_words: int | None = None
) -> list[dict]:
    # Same prompt the production enrichment call site uses, so results are
    # directly comparable to what's already running in the pipeline.
    # max_summary_words=None (default) is the same unbounded prompt as
    # before this parameter existed.
    prompt_text = _build_enhancement_prompt(chunk.text, chunk.tables, max_summary_words)

    if not vision or not chunk.images_b64:
        return [{"role": "user", "content": prompt_text}]

    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for b64 in chunk.images_b64:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    return [{"role": "user", "content": content}]


def _detect_reasoning_truncation(response) -> Optional[str]:
    """Detect a response cut off by max_tokens before producing real content.

    Some providers (confirmed on NVIDIA/Nemotron) return chain-of-thought
    in a separate ``reasoning_content`` field instead of inline <think>
    tags. When max_tokens runs out early enough that generation is still
    inside the reasoning phase, ``content`` doesn't hold a real answer at
    all -- it's empty, or it's a verbatim copy of ``reasoning_content``
    (confirmed empirically: at a low max_tokens the two fields were
    byte-identical). Trusting ``content`` in that case silently returns
    unfinished reasoning as if it were the model's real output.

    Returns an error message if this condition is detected, else ``None``.
    """
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) != "length":
        return None

    reasoning_content = (getattr(choice.message, "reasoning_content", None) or "").strip()
    if not reasoning_content:
        return None  # provider doesn't use this field -- nothing to compare

    content = (choice.message.content or "").strip()
    if not content:
        return (
            "response truncated by max_tokens while still in the reasoning "
            f"phase -- no real content was produced (only reasoning_content, "
            f"{len(reasoning_content)} chars)"
        )

    if content == reasoning_content or reasoning_content.startswith(content):
        return (
            "response truncated by max_tokens before finishing reasoning -- "
            "'content' is just unfinished reasoning, not a real answer"
        )

    return None


async def call_one_model(
    model_row: dict,
    chunk: TestChunk,
    messages: list[dict] | None = None,
    langsmith_metadata: dict | None = None,
    max_summary_words: int | None = None,
) -> ModelResult:
    """Call one registered model and capture tokens/cost/latency/output.

    ``messages`` lets a caller supply a pre-built prompt (e.g. a RAG
    question+context prompt) instead of the default datasheet-summary
    prompt built from ``chunk``. When omitted, behavior is unchanged from
    before this parameter existed.

    ``langsmith_metadata`` (see :func:`build_langsmith_metadata`) is passed
    to LiteLLM as-is; harmless to pass even when tracing is disabled.

    ``max_summary_words`` only affects the default (no ``messages``
    supplied) summary prompt -- see ``_build_enhancement_prompt``. Ignored
    when ``messages`` is supplied directly.
    """
    result = ModelResult(
        model_row_id=model_row["id"],
        model_label=model_row["label"],
        provider=model_row["provider"],
        model_id=model_row["model_id"],
    )

    model_string = _litellm_model_string(model_row["provider"], model_row["model_id"])
    messages = messages or _build_messages(
        chunk, vision=bool(model_row["vision"]), max_summary_words=max_summary_words
    )

    kwargs: dict = dict(
        model=model_string,
        messages=messages,
        api_key=model_row["api_key"],
        temperature=0.0,
        max_tokens=1024,
        timeout=90,
    )
    if model_row.get("base_url"):
        kwargs["api_base"] = model_row["base_url"]
    if langsmith_metadata is not None:
        kwargs["metadata"] = langsmith_metadata

    start = time.perf_counter()
    try:
        response = await litellm.acompletion(**kwargs)
        result.latency_ms = (time.perf_counter() - start) * 1000

        usage = getattr(response, "usage", None)
        if usage:
            result.prompt_tokens = getattr(usage, "prompt_tokens", None)
            result.completion_tokens = getattr(usage, "completion_tokens", None)
            result.total_tokens = getattr(usage, "total_tokens", None)

            # OpenAI-style shape, which LiteLLM normalizes for providers that
            # support it. Empirically confirmed (2026-08-15) that neither
            # Groq nor NVIDIA NIM populate this for the models registered
            # here (completion_tokens_details is None on both) -- left as
            # None/NULL rather than guessed, per this field's own contract:
            # a provider not reporting reasoning tokens is not the same as
            # "confirmed zero reasoning tokens".
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                result.reasoning_tokens = getattr(details, "reasoning_tokens", None)

        result.output_text = response.choices[0].message.content

        result.cost_usd = _compute_cost(response, model_row, result, model_string)

        truncation_error = _detect_reasoning_truncation(response)
        if truncation_error:
            # Real tokens were spent and are worth keeping for the record,
            # but flag this clearly so it's excluded from judge scoring
            # (the scoring callers already skip any result with .error set)
            # instead of silently grading unfinished reasoning as an answer.
            result.error = truncation_error
            logger.warning("Model call for %s: %s", model_row["label"], truncation_error)

    except Exception as e:  # one bad key / dead endpoint shouldn't kill the batch
        result.latency_ms = (time.perf_counter() - start) * 1000
        result.error = str(e)
        logger.warning("Model call failed for %s: %s", model_row["label"], e)

    return result


def _compute_cost(response, model_row: dict, result: ModelResult, model_string: str) -> Optional[float]:
    # Prefer LiteLLM's built-in pricing table -- it covers most named
    # provider/model combos out of the box. Pass ``model`` explicitly: some
    # providers (e.g. Groq) echo back their own bare model string on the
    # response object (no "groq/" prefix), which LiteLLM can't map to a
    # provider on its own, so relying on completion_response alone silently
    # loses pricing that's actually in its table under the prefixed name.
    try:
        cost = litellm.completion_cost(completion_response=response, model=model_string)
        if cost:
            return float(cost)
    except Exception as e:
        logger.warning(
            "completion_cost() failed for %s (%s: %s) -- falling back to manual override",
            model_string, type(e).__name__, e,
        )

    # Fall back to a manual price override entered when the model was
    # registered (needed for brand new / self-hosted / uncatalogued models).
    price_in = model_row.get("price_in_per_1m")
    price_out = model_row.get("price_out_per_1m")
    if price_in is not None and price_out is not None and result.prompt_tokens is not None:
        return (
            (result.prompt_tokens or 0) * price_in
            + (result.completion_tokens or 0) * price_out
        ) / 1_000_000
    return None


async def run_parallel_test(
    model_rows: list[dict],
    chunk: TestChunk,
    *,
    run_id: str | None = None,
    call_site: str = "summary",
    datasheet_source: str = "",
    tested_by: str = "unknown",
    max_summary_words: int | None = None,
) -> list[ModelResult]:
    """Fan ``chunk`` out to every model row in parallel.

    ``run_id`` (reused as the LangSmith ``session_id``) groups every model
    call in this batch into one LangSmith session/trace tree, so the whole
    fan-out is inspectable as a unit. When omitted, no metadata is attached
    (tracing simply has nothing to group by).

    ``max_summary_words`` (see ``_build_enhancement_prompt``) is ``None`` by
    default -- unbounded, unchanged from before this parameter existed.
    """
    tasks = [
        call_one_model(
            row,
            chunk,
            langsmith_metadata=(
                build_langsmith_metadata(
                    session_id=run_id,
                    call_site=call_site,
                    model_label=row["label"],
                    provider=row["provider"],
                    chunk_id=chunk.chunk_id,
                    datasheet_source=datasheet_source,
                    tested_by=tested_by,
                )
                if run_id
                else None
            ),
            max_summary_words=max_summary_words,
        )
        for row in model_rows
    ]
    return await asyncio.gather(*tasks)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
