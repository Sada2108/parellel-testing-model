"""Runtime model registry -- the "paste an API key, no code change" part.

Registered models live in a small SQLite table, not in source code.  The
dashboard's "Model Registry" tab is the only way models get added or
removed; nothing here needs to be edited to test a new model.

``KNOWN_MODELS`` is just a convenience list that pre-fills the add-model
form (provider, base URL, model id, vision support) for models the team is
likely to test, based on the Layer 2 research. Picking "Other / custom" in
the UI skips this and lets someone type in any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent.parent / "model_testing_data" / "model_testing.db"


@dataclass
class KnownModel:
    label: str
    provider: str          # litellm provider prefix, or "custom"
    model_id: str           # the model string the provider expects
    base_url: str            # informational / used when provider == "custom"
    vision: bool
    notes: str = ""


# Seeded from the Layer 2 research (Groq deprecations, Z.ai hosting, HF router
# for GLM-4.5V, Gemini already used as the generation model). Add rows here
# any time -- but the whole point of the registry table is that this list
# is only a shortcut, never a requirement.
KNOWN_MODELS: list[KnownModel] = [
    KnownModel("GPT-OSS-120B (Groq)", "groq", "openai/gpt-oss-120b", "https://api.groq.com/openai/v1", False,
               "Groq's suggested replacement for Llama 3.3 70B Versatile."),
    KnownModel("Qwen3.6-27B (Groq)", "groq", "qwen/qwen3.6-27b", "https://api.groq.com/openai/v1", False,
               "Groq's suggested replacement for Qwen3 32B."),
    KnownModel("Llama 3.3 70B Versatile (Groq)", "groq", "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1", False,
               "Deprecated on Groq free/dev tier -- keep only for a short comparison window."),
    KnownModel("Qwen3 32B (Groq)", "groq", "qwen/qwen3-32b", "https://api.groq.com/openai/v1", False,
               "Deprecated on Groq free/dev tier."),
    KnownModel("GLM-4.5 (Z.ai)", "custom", "glm-4.5", "https://api.z.ai/api/paas/v4", False,
               "Text-only. Not hosted on Groq -- goes through Z.ai's own endpoint."),
    KnownModel("GLM-4.5V (HuggingFace router)", "custom", "zai-org/GLM-4.5V", "https://router.huggingface.co/v1", True,
               "Vision-capable. Currently the pipeline's default ENHANCEMENT_MODEL."),
    KnownModel("Gemini 2.5 Flash", "gemini", "gemini-2.5-flash", "", True,
               "Already used as GENERATION_MODEL for retrieval answers."),
    KnownModel("Gemini 2.5 Flash Lite", "gemini", "gemini-2.5-flash-lite", "", True, ""),
]


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                base_url TEXT,
                api_key TEXT NOT NULL,
                vision INTEGER NOT NULL DEFAULT 0,
                price_in_per_1m REAL,
                price_out_per_1m REAL,
                added_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                tested_by TEXT,
                datasheet_source TEXT,
                chunk_id TEXT,
                model_label TEXT NOT NULL,
                provider TEXT,
                model_id TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd REAL,
                latency_ms REAL,
                quality_score REAL,
                judge_model TEXT,
                output_text TEXT,
                error TEXT
            )
            """
        )


def add_model(
    label: str,
    provider: str,
    model_id: str,
    base_url: str,
    api_key: str,
    vision: bool,
    price_in_per_1m: Optional[float] = None,
    price_out_per_1m: Optional[float] = None,
) -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO models (label, provider, model_id, base_url, api_key, vision,
                                 price_in_per_1m, price_out_per_1m, added_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (label, provider, model_id, base_url, api_key, int(vision),
             price_in_per_1m, price_out_per_1m, time.time()),
        )
        return cur.lastrowid


def list_models() -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM models ORDER BY added_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_model(model_row_id: int) -> Optional[dict]:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_row_id,)).fetchone()
        return dict(row) if row else None


def delete_model(model_row_id: int) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM models WHERE id = ?", (model_row_id,))
