"""Persistent history of every test run -- the leaderboard's data source.

Every model response from every run anyone kicks off gets appended here.
Nothing is ever overwritten, so the leaderboard keeps re-ranking as more
people run more datasheets through more models over time.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .registry import _connect, init_db


def record_result(
    run_id: str,
    created_at: float,
    tested_by: str,
    datasheet_source: str,
    chunk_id: str,
    model_label: str,
    provider: str,
    model_id: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int],
    cost_usd: Optional[float],
    latency_ms: Optional[float],
    quality_score: Optional[float],
    judge_model: Optional[str],
    output_text: Optional[str],
    error: Optional[str] = None,
) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO test_runs (
                run_id, created_at, tested_by, datasheet_source, chunk_id,
                model_label, provider, model_id, prompt_tokens, completion_tokens,
                total_tokens, cost_usd, latency_ms, quality_score, judge_model,
                output_text, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, created_at, tested_by, datasheet_source, chunk_id,
                model_label, provider, model_id, prompt_tokens, completion_tokens,
                total_tokens, cost_usd, latency_ms, quality_score, judge_model,
                output_text, error,
            ),
        )


def all_runs() -> pd.DataFrame:
    init_db()
    with _connect() as conn:
        df = pd.read_sql_query("SELECT * FROM test_runs ORDER BY created_at DESC", conn)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], unit="s")
    return df


def leaderboard() -> pd.DataFrame:
    """Aggregate every recorded run into a per-model leaderboard."""
    df = all_runs()
    if df.empty:
        return df

    ok = df[df["error"].isna() | (df["error"] == "")]
    if ok.empty:
        return pd.DataFrame()

    agg = ok.groupby("model_label").agg(
        runs=("model_label", "count"),
        avg_quality=("quality_score", "mean"),
        avg_tokens=("total_tokens", "mean"),
        avg_cost_usd=("cost_usd", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        total_cost_usd=("cost_usd", "sum"),
    ).reset_index()

    agg = agg.sort_values(by=["avg_quality", "avg_cost_usd"], ascending=[False, True])
    return agg
