"""Parallel Model Test page -- BYOK model registry, parallel run, leaderboard.

Bolted onto the existing dashboard (see ``dashboard.py``) as a new nav entry
rather than a separate app, so it lives next to Chunk Explorer / Compare /
Analytics / Query & Retrieve.

Three tabs:
  - Model Registry: add/remove models by pasting an API key. No code touched.
  - Run Test: pull a datasheet in (upload or URL), pick models, run them all
    in parallel on one chunk, see tokens/cost/latency/quality per model.
  - Leaderboard: every run anyone has ever kicked off, aggregated per model,
    re-ranked as more runs come in.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import api_client
from src.model_testing import harness, registry, scoring, store
from src.model_testing.datasheet_input import TestChunk, fetch_pdf_from_url, load_chunks
from src.model_testing.rag_prompt import build_rag_test_messages
from src.model_testing.registry import KNOWN_MODELS

CUSTOM_LABEL = "Other / custom model"


# ---------------------------------------------------------------------------
# Model Registry tab
# ---------------------------------------------------------------------------
def render_model_registry_tab() -> None:
    st.subheader("Registered models")
    st.caption(
        "Add any model by pasting its API key here -- no code changes needed. "
        "Pick a known model to prefill provider/endpoint, or choose "
        f"\"{CUSTOM_LABEL}\" for anything else."
    )

    with st.form("add_model_form", clear_on_submit=True):
        options = [k.label for k in KNOWN_MODELS] + [CUSTOM_LABEL]
        choice = st.selectbox("Model", options=options, key="add_model_choice")
        known = next((k for k in KNOWN_MODELS if k.label == choice), None)

        col1, col2 = st.columns(2)
        with col1:
            label = st.text_input(
                "Display name",
                value=known.label if known else "",
                placeholder="e.g. My local Qwen3.6-27B",
            )
            provider = st.text_input(
                "Provider (litellm prefix, or 'custom')",
                value=known.provider if known else "custom",
                help="groq / gemini / openai / anthropic / huggingface, or 'custom' "
                     "to call any OpenAI-compatible endpoint via base URL.",
            )
        with col2:
            model_id = st.text_input(
                "Model id",
                value=known.model_id if known else "",
                placeholder="exact model string the provider expects",
            )
            base_url = st.text_input(
                "Base URL (used when provider is 'custom')",
                value=known.base_url if known else "",
                placeholder="https://...",
            )

        vision = st.checkbox(
            "Vision-capable (send datasheet images directly instead of text only)",
            value=known.vision if known else False,
        )
        api_key = st.text_input("API key", type="password")

        with st.expander("Manual pricing override ($ / 1M tokens) -- optional"):
            st.caption(
                "Only needed if LiteLLM doesn't already know this model's price "
                "(new, custom, or self-hosted models). Leave blank to skip cost tracking "
                "for this model."
            )
            pc1, pc2 = st.columns(2)
            price_in = pc1.number_input("Input $/1M tokens", min_value=0.0, value=0.0, step=0.01)
            price_out = pc2.number_input("Output $/1M tokens", min_value=0.0, value=0.0, step=0.01)

        submitted = st.form_submit_button("Add model", type="primary")
        if submitted:
            if not label or not model_id or not api_key:
                st.error("Display name, model id and API key are required.")
            else:
                base_url_clean = base_url.strip()
                looks_like_nvidia_webpage = (
                    "build.nvidia.com" in base_url_clean
                    and "integrate.api.nvidia.com" not in base_url_clean
                )
                if looks_like_nvidia_webpage:
                    st.warning(
                        "This looks like NVIDIA's website URL, not their API endpoint. "
                        "NVIDIA NIM's API base is usually "
                        "https://integrate.api.nvidia.com/v1 -- did you mean that? "
                        "(Saved as entered -- edit the row below if this was a mistake.)"
                    )

                registry.add_model(
                    label=label,
                    provider=provider.strip() or "custom",
                    model_id=model_id.strip(),
                    base_url=base_url_clean,
                    api_key=api_key,
                    vision=vision,
                    price_in_per_1m=price_in or None,
                    price_out_per_1m=price_out or None,
                )
                st.success(f"Added {label}.")
                if not looks_like_nvidia_webpage:
                    st.rerun()

    st.divider()
    models = registry.list_models()
    if not models:
        st.info("No models registered yet -- add one above to get started.")
        return

    st.caption(f"{len(models)} model(s) registered")
    for m in models:
        cols = st.columns([3, 2, 2, 1, 1])
        cols[0].write(f"**{m['label']}**")
        cols[1].write(m["provider"])
        cols[2].write(m["model_id"])
        cols[3].write("Vision" if m["vision"] else "Text")
        if cols[4].button("Remove", key=f"remove_{m['id']}"):
            registry.delete_model(m["id"])
            st.rerun()


# ---------------------------------------------------------------------------
# Run Test tab
# ---------------------------------------------------------------------------
def _init_run_state() -> None:
    defaults = {
        "mt_chunks": None,
        "mt_datasheet_source": None,
        "mt_run_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


async def _run_and_score(
    model_rows: list[dict],
    chunk: TestChunk,
    judge_row: dict | None,
) -> list[harness.ModelResult]:
    results = await harness.run_parallel_test(model_rows, chunk)

    if judge_row is not None:
        score_tasks = []
        scorable = [r for r in results if r.output_text and not r.error]
        for r in scorable:
            score_tasks.append(scoring.score_summary(chunk, r.output_text, judge_row))
        scores = await asyncio.gather(*score_tasks) if score_tasks else []
        for r, (score, raw) in zip(scorable, scores):
            r.quality_score = score  # type: ignore[attr-defined]
            if score is None and raw:
                # Don't let a failed/unparseable judge call vanish silently --
                # surface it via the same error column the UI already reads,
                # without touching the model call's own result/error.
                r.error = f"quality scoring failed: {raw[:2000]}"

    return results


def render_run_tab() -> None:
    """Run Test tab: dispatch to the call site the user picks.

    "Datasheet summary" is the original, untouched flow. "Query & Retrieve"
    is the new call site, added below without modifying the summary path.
    """
    call_site = st.radio(
        "Call site",
        ["Datasheet summary", "Query & Retrieve"],
        horizontal=True,
        key="mt_call_site",
    )
    st.divider()
    if call_site == "Datasheet summary":
        _render_summary_run_tab()
    else:
        _render_query_retrieve_run_tab()


def _render_summary_run_tab() -> None:
    _init_run_state()

    st.subheader("1. Get a datasheet in")
    source_mode = st.radio("Source", ["Upload file", "From URL"], horizontal=True)

    pdf_path: Path | None = None
    source_label = None

    if source_mode == "Upload file":
        uploaded = st.file_uploader("Datasheet PDF", type=["pdf"])
        if uploaded is not None:
            tmp_path = Path("/tmp") / uploaded.name
            tmp_path.write_bytes(uploaded.getvalue())
            pdf_path = tmp_path
            source_label = uploaded.name
    else:
        url = st.text_input("Datasheet URL", placeholder="https://.../datasheet.pdf")
        if url:
            source_label = url

    parse_col, _ = st.columns([1, 4])
    if parse_col.button("Parse datasheet", type="primary", disabled=pdf_path is None and not (source_mode == "From URL" and source_label)):
        with st.spinner("Extracting and chunking..."):
            try:
                if source_mode == "From URL":
                    pdf_path = fetch_pdf_from_url(source_label)
                chunks = load_chunks(pdf_path, max_chunks=50)
                st.session_state.mt_chunks = chunks
                st.session_state.mt_datasheet_source = source_label
                st.success(f"Parsed {len(chunks)} chunk(s).")
            except Exception as e:
                st.error(f"Couldn't parse that datasheet: {e}")

    chunks = st.session_state.mt_chunks
    if not chunks:
        st.info("Parse a datasheet to continue.")
        return

    st.divider()
    st.subheader("2. Pick a chunk")
    chunk_labels = {
        c.chunk_id: f"Chunk {c.chunk_id} -- {c.text[:80].strip()}..." for c in chunks
    }
    chunk_id = st.selectbox(
        "Chunk to test",
        options=list(chunk_labels.keys()),
        format_func=lambda cid: chunk_labels[cid],
    )
    chunk = next(c for c in chunks if c.chunk_id == chunk_id)
    st.caption(f"{len(chunk.tables)} table(s), {len(chunk.images_b64)} image(s) in this chunk.")

    st.divider()
    st.subheader("3. Pick models and run")
    models = registry.list_models()
    if not models:
        st.warning("No models registered yet -- add some in the Model Registry tab first.")
        return

    label_to_row = {m["label"]: m for m in models}
    selected_labels = st.multiselect("Models to test in parallel", options=list(label_to_row.keys()))

    judge_choice = st.selectbox(
        "Judge model for quality scoring (optional -- ideally not one you're testing)",
        options=["No quality scoring"] + list(label_to_row.keys()),
    )
    judge_row = label_to_row.get(judge_choice) if judge_choice != "No quality scoring" else None

    tested_by = st.text_input("Tested by", placeholder="your name")

    if st.button("Run parallel test", type="primary", disabled=not selected_labels):
        model_rows = [label_to_row[lbl] for lbl in selected_labels]
        run_id = harness.new_run_id()

        with st.spinner(f"Running {len(model_rows)} model(s) in parallel..."):
            results = asyncio.run(_run_and_score(model_rows, chunk, judge_row))

        created_at = time.time()
        for r in results:
            quality = getattr(r, "quality_score", None)
            store.record_result(
                run_id=run_id,
                created_at=created_at,
                tested_by=tested_by or "unknown",
                datasheet_source=st.session_state.mt_datasheet_source or "",
                chunk_id=chunk.chunk_id,
                model_label=r.model_label,
                provider=r.provider,
                model_id=r.model_id,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
                total_tokens=r.total_tokens,
                cost_usd=r.cost_usd,
                latency_ms=r.latency_ms,
                quality_score=quality,
                judge_model=judge_row["label"] if judge_row else None,
                output_text=r.output_text,
                error=r.error,
            )

        st.session_state.mt_run_results = results
        st.success("Done -- results below, and saved to the Leaderboard tab.")

    results = st.session_state.mt_run_results
    if results:
        st.divider()
        st.subheader("Results")
        for r in results:
            quality = getattr(r, "quality_score", None)
            header = f"**{r.model_label}**"
            if r.error:
                header += "  --  :red[failed]"
            with st.container(border=True):
                st.markdown(header)
                cols = st.columns(5)
                cols[0].metric("Tokens", r.total_tokens or "-")
                cols[1].metric("Cost", f"${r.cost_usd:.5f}" if r.cost_usd is not None else "-")
                cols[2].metric("Latency", f"{r.latency_ms:.0f} ms" if r.latency_ms else "-")
                cols[3].metric("Quality", f"{quality:.0f}/100" if quality is not None else "-")
                cols[4].write("")
                if r.error:
                    st.error(r.error)
                elif r.output_text:
                    with st.expander("Output"):
                        st.markdown(r.output_text)


# ---------------------------------------------------------------------------
# Run Test tab -- Query & Retrieve call site
# ---------------------------------------------------------------------------
def _init_qr_state() -> None:
    defaults = {
        "mt_qr_chunks": None,
        "mt_qr_question": None,
        "mt_qr_run_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


async def _run_and_score_rag(
    model_rows: list[dict],
    question: str,
    chunks: list[dict],
    judge_row: dict | None,
) -> list[harness.ModelResult]:
    """Fan a RAG question+context prompt out to every selected model.

    Unlike the summary path, the prompt differs per model (vision models
    get chunk images attached, text-only models don't), so this builds
    ``messages`` per model row and passes them into ``call_one_model``
    instead of relying on its default chunk-derived prompt.
    """
    tasks = []
    for row in model_rows:
        messages = build_rag_test_messages(question, chunks, vision=bool(row["vision"]))
        dummy_chunk = TestChunk(chunk_id="query", text=question)
        tasks.append(harness.call_one_model(row, dummy_chunk, messages=messages))
    results = await asyncio.gather(*tasks)

    if judge_row is not None:
        context_text = "\n\n".join((c.get("enhanced_content") or "") for c in chunks)
        score_tasks = []
        scorable = [r for r in results if r.output_text and not r.error]
        for r in scorable:
            score_tasks.append(scoring.score_answer(question, context_text, r.output_text, judge_row))
        scores = await asyncio.gather(*score_tasks) if score_tasks else []
        for r, (score, raw) in zip(scorable, scores):
            r.quality_score = score  # type: ignore[attr-defined]
            if score is None and raw:
                r.error = f"quality scoring failed: {raw[:2000]}"

    return results


def _render_query_retrieve_run_tab() -> None:
    _init_qr_state()

    st.subheader("1. Ask a question")
    question = st.text_input(
        "Question", placeholder="e.g. What is the pin configuration of the LM2596?"
    )

    retrieval_mode = st.radio(
        "Context source",
        ["Retrieve via backend", "Paste context manually"],
        horizontal=True,
    )

    if retrieval_mode == "Retrieve via backend":
        top_k = st.number_input("Chunks to retrieve", min_value=1, max_value=30, value=10)
        if st.button("Retrieve chunks", type="primary", disabled=not question):
            try:
                with st.spinner("Retrieving from backend..."):
                    response = api_client.retrieve(question, top_k=int(top_k))
                st.session_state.mt_qr_chunks = response["chunks"]
                st.session_state.mt_qr_question = question
                st.success(f"Retrieved {len(response['chunks'])} chunk(s).")
            except Exception as e:
                st.error(
                    f"Couldn't reach the backend: {e}\n\n"
                    "Make sure the FastAPI backend is running "
                    "(uvicorn api.main:app --port 8000), or switch to "
                    '"Paste context manually" below.'
                )
    else:
        pasted = st.text_area(
            "Context", height=200, placeholder="Paste retrieved context / chunk text here..."
        )
        if pasted and question:
            st.session_state.mt_qr_chunks = [
                {
                    "enhanced_content": pasted,
                    "raw_text": pasted,
                    "tables_html": [],
                    "images_base64": [],
                }
            ]
            st.session_state.mt_qr_question = question

    chunks = st.session_state.mt_qr_chunks
    if not chunks or not st.session_state.mt_qr_question:
        st.info("Enter a question and get context (via backend or pasted) to continue.")
        return

    question = st.session_state.mt_qr_question
    st.caption(f"{len(chunks)} chunk(s) of context ready.")

    st.divider()
    st.subheader("2. Pick models and run")
    models = registry.list_models()
    if not models:
        st.warning("No models registered yet -- add some in the Model Registry tab first.")
        return

    label_to_row = {m["label"]: m for m in models}
    selected_labels = st.multiselect(
        "Models to test in parallel", options=list(label_to_row.keys()), key="qr_models"
    )

    judge_choice = st.selectbox(
        "Judge model for quality scoring (optional -- ideally not one you're testing)",
        options=["No quality scoring"] + list(label_to_row.keys()),
        key="qr_judge",
    )
    judge_row = label_to_row.get(judge_choice) if judge_choice != "No quality scoring" else None

    tested_by = st.text_input("Tested by", placeholder="your name", key="qr_tested_by")

    if st.button(
        "Run parallel test", type="primary", disabled=not selected_labels, key="qr_run_button"
    ):
        model_rows = [label_to_row[lbl] for lbl in selected_labels]
        run_id = harness.new_run_id()

        with st.spinner(f"Running {len(model_rows)} model(s) in parallel..."):
            results = asyncio.run(_run_and_score_rag(model_rows, question, chunks, judge_row))

        created_at = time.time()
        for r in results:
            quality = getattr(r, "quality_score", None)
            store.record_result(
                run_id=run_id,
                created_at=created_at,
                tested_by=tested_by or "unknown",
                datasheet_source=question,
                chunk_id=f"qr:{len(chunks)}chunks",
                model_label=r.model_label,
                provider=r.provider,
                model_id=r.model_id,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
                total_tokens=r.total_tokens,
                cost_usd=r.cost_usd,
                latency_ms=r.latency_ms,
                quality_score=quality,
                judge_model=judge_row["label"] if judge_row else None,
                output_text=r.output_text,
                error=r.error,
                call_site="query_retrieve",
            )

        st.session_state.mt_qr_run_results = results
        st.success("Done -- results below, and saved to the Leaderboard tab.")

    results = st.session_state.mt_qr_run_results
    if results:
        st.divider()
        st.subheader("Results")
        for r in results:
            quality = getattr(r, "quality_score", None)
            header = f"**{r.model_label}**"
            if r.error:
                header += "  --  :red[failed]"
            with st.container(border=True):
                st.markdown(header)
                cols = st.columns(5)
                cols[0].metric("Tokens", r.total_tokens or "-")
                cols[1].metric("Cost", f"${r.cost_usd:.5f}" if r.cost_usd is not None else "-")
                cols[2].metric("Latency", f"{r.latency_ms:.0f} ms" if r.latency_ms else "-")
                cols[3].metric("Quality", f"{quality:.0f}/100" if quality is not None else "-")
                cols[4].write("")
                if r.error:
                    st.error(r.error)
                elif r.output_text:
                    with st.expander("Output"):
                        st.markdown(r.output_text)


# ---------------------------------------------------------------------------
# Leaderboard tab
# ---------------------------------------------------------------------------
_CALL_SITE_LABELS = {"summary": "Datasheet summary", "query_retrieve": "Query & Retrieve"}


def render_leaderboard_tab() -> None:
    st.subheader("Leaderboard")
    call_site_label = st.radio(
        "Call site",
        list(_CALL_SITE_LABELS.values()),
        horizontal=True,
        key="lb_call_site",
    )
    call_site = next(k for k, v in _CALL_SITE_LABELS.items() if v == call_site_label)

    board = store.leaderboard(call_site=call_site)
    st.caption(
        f"Aggregated across every \"{call_site_label}\" run anyone has kicked off -- "
        "ranked by quality, then cost. Summary and Query & Retrieve runs measure "
        "different tasks, so they're ranked separately rather than blended together."
    )

    if board.empty:
        st.info("No completed runs yet.")
    else:
        st.dataframe(
            board,
            width="stretch",
            hide_index=True,
            column_config={
                "model_label": "Model",
                "runs": st.column_config.NumberColumn("Runs"),
                "avg_quality": st.column_config.ProgressColumn(
                    "Avg quality", min_value=0, max_value=100, format="%.0f"
                ),
                "avg_tokens": st.column_config.NumberColumn("Avg tokens", format="%.0f"),
                "avg_cost_usd": st.column_config.NumberColumn("Avg cost ($)", format="%.5f"),
                "avg_latency_ms": st.column_config.NumberColumn("Avg latency (ms)", format="%.0f"),
                "total_cost_usd": st.column_config.NumberColumn("Total spent ($)", format="%.4f"),
            },
        )

        # Quality is the axis that actually needs a real value to plot
        # meaningfully -- a model with no judge score yet can't be placed
        # on this chart. Cost is different: missing cost data shouldn't
        # make a model's point vanish, since $0 is not what "unknown" looks
        # like. So: drop rows with no quality score, but for rows that DO
        # have quality, plot missing cost as $0 and mark it clearly instead
        # of silently dropping the whole point (which is what a bare
        # px.scatter does with any NaN in x or y -- that's what was making
        # this chart render completely empty even with real runs recorded).
        plottable = board[board["avg_quality"].notna()].copy()
        cost_missing = plottable["avg_cost_usd"].isna()
        plottable["avg_cost_usd"] = plottable["avg_cost_usd"].fillna(0.0)
        plottable["Cost data"] = cost_missing.map({True: "no cost data (plotted at $0)", False: "cost known"})

        if plottable.empty:
            st.info("No cost/quality data yet for these runs -- run a test with a judge model selected to populate this chart.")
        else:
            import plotly.express as px

            fig = px.scatter(
                plottable,
                x="avg_cost_usd",
                y="avg_quality",
                size="runs",
                text="model_label",
                color="Cost data",
                color_discrete_map={"cost known": "#636EFA", "no cost data (plotted at $0)": "#EF553B"},
                title="Cost vs. quality (bottom-right is the sweet spot)",
                labels={"avg_cost_usd": "Avg cost per call ($)", "avg_quality": "Avg quality score"},
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, width="stretch")
            if cost_missing.any():
                st.caption(
                    "Points in red have no cost data (LiteLLM doesn't know this model's "
                    "price and no manual override is set) -- plotted at $0, not a real cost."
                )

    st.divider()
    st.subheader("Full run history")
    runs = store.all_runs()
    if runs.empty:
        st.info("Nothing recorded yet.")
        return
    runs = runs[runs["call_site"] == call_site]
    if runs.empty:
        st.info(f'No "{call_site_label}" runs recorded yet.')
        return

    display_cols = [
        "created_at", "tested_by", "datasheet_source", "chunk_id", "model_label",
        "total_tokens", "cost_usd", "latency_ms", "quality_score", "error",
    ]
    st.dataframe(runs[display_cols], width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
def render() -> None:
    st.header("Parallel Model Test")
    tab1, tab2, tab3 = st.tabs(["Model Registry", "Run Test", "Leaderboard"])
    with tab1:
        render_model_registry_tab()
    with tab2:
        render_run_tab()
    with tab3:
        render_leaderboard_tab()
