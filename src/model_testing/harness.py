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
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import litellm

from src.ingestion.enrich import _build_enhancement_prompt
from src.logger import get_logger

from .datasheet_input import TestChunk

logger = get_logger(__name__)

# LiteLLM is chatty by default about missing prices / unmapped models --
# those aren't errors here, they're expected for brand new or custom models.
litellm.suppress_debug_info = True


@dataclass
class ModelResult:
    model_row_id: int
    model_label: str
    provider: str
    model_id: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
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


def _build_messages(chunk: TestChunk, vision: bool) -> list[dict]:
    # Same prompt the production enrichment call site uses, so results are
    # directly comparable to what's already running in the pipeline.
    prompt_text = _build_enhancement_prompt(chunk.text, chunk.tables)

    if not vision or not chunk.images_b64:
        return [{"role": "user", "content": prompt_text}]

    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for b64 in chunk.images_b64:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    return [{"role": "user", "content": content}]


async def call_one_model(
    model_row: dict, chunk: TestChunk, messages: list[dict] | None = None
) -> ModelResult:
    """Call one registered model and capture tokens/cost/latency/output.

    ``messages`` lets a caller supply a pre-built prompt (e.g. a RAG
    question+context prompt) instead of the default datasheet-summary
    prompt built from ``chunk``. When omitted, behavior is unchanged from
    before this parameter existed.
    """
    result = ModelResult(
        model_row_id=model_row["id"],
        model_label=model_row["label"],
        provider=model_row["provider"],
        model_id=model_row["model_id"],
    )

    model_string = _litellm_model_string(model_row["provider"], model_row["model_id"])
    messages = messages or _build_messages(chunk, vision=bool(model_row["vision"]))

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

    start = time.perf_counter()
    try:
        response = await litellm.acompletion(**kwargs)
        result.latency_ms = (time.perf_counter() - start) * 1000

        usage = getattr(response, "usage", None)
        if usage:
            result.prompt_tokens = getattr(usage, "prompt_tokens", None)
            result.completion_tokens = getattr(usage, "completion_tokens", None)
            result.total_tokens = getattr(usage, "total_tokens", None)

        result.output_text = response.choices[0].message.content

        result.cost_usd = _compute_cost(response, model_row, result, model_string)

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
    except Exception:
        pass

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


async def run_parallel_test(model_rows: list[dict], chunk: TestChunk) -> list[ModelResult]:
    tasks = [call_one_model(row, chunk) for row in model_rows]
    return await asyncio.gather(*tasks)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
