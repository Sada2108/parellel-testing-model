"""In-memory image store backing the ``/images/{response_id}/{index}`` URLs.

Images are decoded from base64 when a response is built and kept in memory
briefly so HTTP clients can fetch them by URL.  Entries expire after a TTL
and the registry is capped to bound memory usage.
"""

from __future__ import annotations

import base64
import threading
import time
import uuid

_IMAGE_TTL_SECONDS = 30 * 60
_MAX_ENTRIES = 256


class ImageRegistry:
    """Thread-safe registry mapping ``response_id -> list[image bytes]``."""

    def __init__(self, ttl_seconds: int = _IMAGE_TTL_SECONDS, max_entries: int = _MAX_ENTRIES):
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._items: dict[str, list[bytes]] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()

    def register(self, images_by_chunk: list[list[str]]) -> str:
        """Store the images of every chunk and return a fresh response id.

        Args:
            images_by_chunk: For each chunk, its list of base64 images.

        Returns:
            The response id to use in ``/images/{response_id}/{index}`` URLs.
        """
        with self._lock:
            flat: list[bytes] = []
            for images in images_by_chunk:
                for b64 in images:
                    try:
                        flat.append(base64.b64decode(b64))
                    except Exception:
                        continue
            response_id = uuid.uuid4().hex
            self._items[response_id] = flat
            self._timestamps[response_id] = time.monotonic()
            self._prune()
            return response_id

    def get(self, response_id: str, index: int) -> bytes | None:
        """Return the raw image bytes for a response, or None if missing/expired."""
        with self._lock:
            images = self._items.get(response_id)
            if images is None:
                return None
            if 0 <= index < len(images):
                return images[index]
            return None

    def _prune(self) -> None:
        now = time.monotonic()
        expired = [k for k, t in self._timestamps.items() if now - t > self._ttl]
        for key in expired:
            self._items.pop(key, None)
            self._timestamps.pop(key, None)
        while len(self._items) > self._max_entries:
            oldest = min(self._timestamps, key=self._timestamps.get)
            self._items.pop(oldest, None)
            self._timestamps.pop(oldest, None)
