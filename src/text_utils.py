"""Small text-cleanup helpers shared across the ingestion pipeline and the
model testing harness. Kept dependency-free (no litellm/openai imports) so
either side can import it without pulling in the other.
"""

from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output.

    Reasoning models (Qwen especially, confirmed) sometimes prefix their
    real output with a <think> block. A no-op for content that never had
    one -- verified empirically against a real call to the production
    enhancement model (GLM-4.5V via HF router), which currently returns
    plain text with no <think> tags.

    Handles multiple blocks (all removed) and an unclosed <think> tag: if
    generation got cut off while still inside the reasoning block, there's
    no </think> to match, so everything from <think> to the end of the
    string is treated as reasoning too -- a model that ran out of budget
    mid-thought never produced real content after that point.
    """
    stripped = _THINK_BLOCK_RE.sub("", text)
    stripped = _UNCLOSED_THINK_RE.sub("", stripped)
    return stripped.strip()
