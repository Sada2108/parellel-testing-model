"""Quality scoring for the enrichment call site's actual output: a summary.

Deliberately not a fixed extraction schema (component_values / pin_count /
etc.) -- that's not what this call site produces. ``enrich.py`` asks each
model for a free-text *searchable description* of a chunk (facts, topics,
questions it could answer, visual content, alternative search terms), so
quality here means "how good is that summary", not "did every field parse".

Scored by a judge model against the original chunk content, on a 0-100
scale. Pick a judge model that isn't also being tested, so nothing is
grading its own homework.
"""

from __future__ import annotations

import re
from typing import Optional

import litellm

from .datasheet_input import TestChunk
from .harness import _litellm_model_string

_SCORE_RE = re.compile(r"SCORE\s*:\s*(\d{1,3})", re.IGNORECASE)

JUDGE_PROMPT_TEMPLATE = """You are grading a searchable summary generated for a chunk of an \
electronics datasheet. Score how well the SUMMARY captures the ORIGINAL CONTENT.

Score 0-100 on these dimensions, weighted roughly evenly:
- Key facts and numbers preserved (values, ratings, part numbers)
- Coverage of the main topics/sections present in the original
- Whether the summary would actually help someone search for this content
- No fabricated details that aren't supported by the original content

ORIGINAL TEXT:
{original_text}

ORIGINAL TABLES (HTML):
{original_tables}

SUMMARY TO GRADE:
{summary}

Respond with a short rationale (2-3 sentences), then end your response on its own \
line with exactly:
SCORE: <integer 0-100>
"""


async def score_summary(
    chunk: TestChunk,
    summary: str,
    judge_model_row: dict,
) -> tuple[Optional[float], Optional[str]]:
    """Returns (score, raw_judge_response). Score is None if grading failed
    or the judge's response couldn't be parsed."""
    if not summary:
        return None, None

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        original_text=chunk.text[:4000],
        original_tables="\n\n".join(chunk.tables)[:2000] or "(none)",
        summary=summary[:4000],
    )

    model_string = _litellm_model_string(judge_model_row["provider"], judge_model_row["model_id"])
    kwargs: dict = dict(
        model=model_string,
        messages=[{"role": "user", "content": prompt}],
        api_key=judge_model_row["api_key"],
        temperature=0.0,
        # Judge models are often reasoning models that spend hundreds of
        # tokens on a <think> block before ever writing "SCORE: N" -- 300
        # was cutting them off mid-thought every time, well before the
        # score line. Give enough headroom for that plus the rationale.
        max_tokens=2048,
        timeout=90,
    )
    if judge_model_row.get("base_url"):
        kwargs["api_base"] = judge_model_row["base_url"]

    try:
        response = await litellm.acompletion(**kwargs)
        text = response.choices[0].message.content or ""
    except Exception as e:
        return None, f"[judge call raised {type(e).__name__}] {e}"

    match = _SCORE_RE.search(text)
    if not match:
        return None, f"[no SCORE line found in judge response] {text}"

    score = float(match.group(1))
    return max(0.0, min(100.0, score)), text
