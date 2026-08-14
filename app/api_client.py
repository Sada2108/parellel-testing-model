"""Thin HTTP client for the FastAPI backend (``api.main``).

Lets the Streamlit dashboard talk to the API instead of loading the vector
store and calling the LLM itself.  Image URLs returned by the backend are
relative (``/images/{response_id}/{index}``); this client rewrites them to
absolute URLs so ``st.image`` can load them directly.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import requests

from config.settings import API_BASE_URL

_REQUEST_TIMEOUT = 300


def _url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}"


def _absolutize_image_urls(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn relative ``/images/...`` URLs into absolute backend URLs."""
    for chunk in payload.get("chunks", []):
        chunk["image_urls"] = [_url(u) for u in chunk.get("image_urls", [])]
    return payload


def health() -> dict:
    """Return the backend health payload, raising on failure/unreachable."""
    response = requests.get(_url("/health"), timeout=10)
    response.raise_for_status()
    return response.json()


def retrieve(query: str, top_k: int = 10) -> dict:
    """Retrieve chunks only (no answer generation)."""
    response = requests.post(
        _url("/retrieve"),
        json={"query": query, "top_k": top_k},
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return _absolutize_image_urls(response.json())


def query(query: str, top_k: int = 10) -> dict:
    """Retrieve chunks and generate the full answer in one call."""
    response = requests.post(
        _url("/query"),
        json={"query": query, "top_k": top_k},
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return _absolutize_image_urls(response.json())


def stream(query: str, top_k: int = 10) -> Iterator[tuple[str, dict]]:
    """Stream SSE events from ``/query/stream``.

    Yields ``(event, data)`` pairs where ``event`` is one of
    ``retrieval``, ``token``, ``done`` or ``error``.
    """
    with requests.get(
        _url("/query/stream"),
        params={"query": query, "top_k": top_k},
        stream=True,
        timeout=_REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()
        event: str | None = None
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                if event == "retrieval":
                    data = _absolutize_image_urls(data)
                yield event, data
