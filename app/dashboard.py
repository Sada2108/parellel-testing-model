"""Multimodal RAG Chunk Inspector -- Debug and Validate Chunks."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import base64
import io
import time
from pathlib import Path
from typing import Any, Optional

import streamlit as st


def _merge_streamlit_secrets() -> None:
    """Expose Streamlit secrets as environment variables.

    Locally the app reads ``.env`` (gitignored).  On Streamlit Cloud the
    equivalent is ``st.secrets``, but not every runtime surfaces those as
    process environment variables for third-party SDKs (LangSmith, Hugging
    Face, Gemini).  Copying them into ``os.environ`` before the settings and
    tracing modules are imported guarantees all backends read the same config
    in both environments.
    """
    try:
        for key, value in st.secrets.items():
            os.environ.setdefault(str(key), str(value))
    except Exception:
        pass

_merge_streamlit_secrets()

import pandas as pd
from PIL import Image
import plotly.express as px

from config.settings import API_BASE_URL
from src.logger import get_logger

from api_client import health as api_health
from api_client import stream as api_stream
import model_test_page

logger = get_logger(__name__)

PAGE_TITLE = "Multimodal RAG Chunk Inspector"
LAYOUT = "wide"


def log_tracing_status() -> None:
    """Log whether LangSmith tracing is enabled in this runtime.

    ``@traceable`` silently becomes a no-op when tracing is not configured, so
    the app still runs but produces no traces.  These lines make the Cloud
    startup logs show exactly what the SDK sees.
    """
    flag = os.getenv("LANGSMITH_TRACING_V2") or os.getenv("LANGCHAIN_TRACING_V2")
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    project = (
        os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "default"
    )
    background = os.getenv("LANGSMITH_TRACING_BACKGROUND", "true")

    logger.info(
        "LangSmith tracing: enabled=%s, api_key_set=%s, project=%r, "
        "background_post=%s",
        bool(flag and str(flag).lower() == "true"),
        bool(api_key),
        project,
        background,
    )
    if not (flag and str(flag).lower() == "true") or not api_key:
        logger.warning(
            "LangSmith tracing appears DISABLED. Set LANGSMITH_TRACING_V2=true "
            "and LANGSMITH_API_KEY (or LANGCHAIN_* equivalents) in Streamlit "
            "secrets, then confirm the project name matches the one you view."
        )


def init_state() -> None:
    defaults = {
        "data": None,
        "file_name": None,
        "filtered_indices": None,
        "selected_chunk_id": None,
        "search_query": "",
        "has_images": False,
        "has_tables": False,
        "has_raw_text": False,
        "has_enhanced": False,
        "chunk_id_filter": "",
        "min_len": 0,
        "max_len": 100_000,
        "compare_a": None,
        "compare_b": None,
        "chat_query": "",
        "chat_answer": None,
        "chat_retrieval": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_data
def load_json(file: Any) -> list[dict]:
    return json.load(file)


def load_json_from_path(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files() -> list[str]:
    json_dir = Path(__file__).resolve().parent.parent / "json"
    if not json_dir.exists():
        return []
    return sorted(f.name for f in json_dir.glob("*.json"))


def _oc(meta: dict) -> dict:
    return meta.get("original_content") or {}


def get_images_base64(chunk: dict) -> list[str]:
    return _oc(chunk.get("metadata", {})).get("images_base64") or []


def get_tables_html(chunk: dict) -> list[str]:
    return _oc(chunk.get("metadata", {})).get("tables_html") or []


def get_raw_text(chunk: dict) -> str:
    return _oc(chunk.get("metadata", {})).get("raw_text") or ""


def get_enhanced_content(chunk: dict) -> str:
    return chunk.get("enhanced_content") or ""


def word_count(text: str) -> int:
    return len(text.split())


def health_score(chunk: dict) -> tuple[int, str]:
    score = 0
    if get_enhanced_content(chunk):
        score += 25
    if get_raw_text(chunk):
        score += 25
    if get_images_base64(chunk):
        score += 25
    if get_tables_html(chunk):
        score += 25

    if score >= 75:
        label = "Pass"
    elif score >= 50:
        label = "Partial"
    else:
        label = "Fail"
    return score, label


def render_image_from_base64(b64_str: str) -> Optional[Image.Image]:
    try:
        raw = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(raw))
    except Exception:
        return None


def parse_table_html(html: str) -> Optional[pd.DataFrame]:
    try:
        tables = pd.read_html(io.StringIO(html))
        if tables:
            return tables[0]
    except Exception:
        pass
    return None


def apply_filters(data: list[dict]) -> list[dict]:
    result = list(data)

    q = st.session_state.search_query.strip().lower()
    if q:
        filtered = []
        for c in result:
            haystack = (
                get_enhanced_content(c) + " " + get_raw_text(c)
            ).lower()
            if q in haystack:
                filtered.append(c)
        result = filtered

    if st.session_state.has_images:
        result = [c for c in result if get_images_base64(c)]
    if st.session_state.has_tables:
        result = [c for c in result if get_tables_html(c)]
    if st.session_state.has_raw_text:
        result = [c for c in result if get_raw_text(c)]
    if st.session_state.has_enhanced:
        result = [c for c in result if get_enhanced_content(c)]

    cid = st.session_state.chunk_id_filter.strip()
    if cid:
        try:
            num = int(cid)
            result = [c for c in result if c.get("chunk_id") == num]
        except ValueError:
            pass

    min_l = st.session_state.min_len
    max_l = st.session_state.max_len
    result = [
        c for c in result if min_l <= len(get_raw_text(c)) <= max_l
    ]

    return result


def _on_filter_change() -> None:
    st.session_state.selected_chunk_id = None
    st.session_state.filtered_indices = None


def render_sidebar(show_data_section: bool = True) -> None:
    with st.sidebar:
        if not show_data_section:
            return
        st.header("Data")

        json_files = list_json_files()
        if json_files:
            selected_file = st.selectbox(
                "Select a datasheet",
                options=[""] + json_files,
                format_func=lambda x: "Choose a file..." if x == "" else x,
                key="json_selectbox",
            )
            if selected_file:
                json_path = Path(__file__).resolve().parent.parent / "json" / selected_file
                if selected_file != st.session_state.get("file_name"):
                    st.session_state.data = load_json_from_path(str(json_path))
                    st.session_state.file_name = selected_file
                    st.session_state.filtered_indices = None
                    st.session_state.selected_chunk_id = None
                    st.rerun()

        st.divider()
        st.caption("Or upload your own JSON")
        uploaded = st.file_uploader(
            "Upload JSON", type=["json"], label_visibility="collapsed"
        )

        if uploaded and uploaded.name != st.session_state.get("file_name"):
            st.session_state.data = load_json(uploaded)
            st.session_state.file_name = uploaded.name
            st.session_state.filtered_indices = None
            st.session_state.selected_chunk_id = None
            st.rerun()

        data = st.session_state.data
        if data is not None:
            st.metric("Total Chunks", len(data))
            st.divider()

            st.text_input(
                "Search",
                key="search_query",
                placeholder="Search enhanced / raw text ...",
                on_change=_on_filter_change,
            )

            st.divider()
            st.subheader("Filters")

            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Has Images", key="has_images", on_change=_on_filter_change)
                st.checkbox("Has Tables", key="has_tables", on_change=_on_filter_change)
            with col2:
                st.checkbox("Has Raw Text", key="has_raw_text", on_change=_on_filter_change)
                st.checkbox("Has Enhanced", key="has_enhanced", on_change=_on_filter_change)

            st.text_input("Chunk ID", key="chunk_id_filter", on_change=_on_filter_change)

            st.number_input(
                "Min raw text length",
                min_value=0,
                max_value=100_000,
                value=0,
                key="min_len",
                on_change=_on_filter_change,
            )
            st.number_input(
                "Max raw text length",
                min_value=0,
                max_value=100_000,
                value=100_000,
                key="max_len",
                on_change=_on_filter_change,
            )


def render_dashboard(data: list[dict]) -> None:
    total = len(data)
    total_images = sum(len(get_images_base64(c)) for c in data)
    total_tables = sum(len(get_tables_html(c)) for c in data)
    chunks_with_images = sum(1 for c in data if get_images_base64(c))
    chunks_with_tables = sum(1 for c in data if get_tables_html(c))
    avg_raw = (
        sum(len(get_raw_text(c)) for c in data) / total if total else 0
    )
    avg_enh = (
        sum(len(get_enhanced_content(c)) for c in data) / total if total else 0
    )

    cols = st.columns(7)
    metrics = [
        ("Total Chunks", total),
        ("Total Images", total_images),
        ("Total Tables", total_tables),
        ("Chunks w/ Images", chunks_with_images),
        ("Chunks w/ Tables", chunks_with_tables),
        ("Avg Raw Length", f"{avg_raw:.0f}"),
        ("Avg Enhanced Length", f"{avg_enh:.0f}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def build_chunk_df(data: list[dict]) -> pd.DataFrame:
    rows = []
    for c in data:
        cid = c.get("chunk_id", "?")
        n_img = len(get_images_base64(c))
        n_tbl = len(get_tables_html(c))
        rlen = len(get_raw_text(c))
        elen = len(get_enhanced_content(c))
        score, _ = health_score(c)
        rows.append(
            {
                "Chunk ID": cid,
                "Images": n_img,
                "Tables": n_tbl,
                "Raw Len": rlen,
                "Enh Len": elen,
                "Health": score,
            }
        )
    return pd.DataFrame(rows)


def render_chunk_list(data: list[dict]) -> None:
    st.subheader("Chunks")
    df = build_chunk_df(data)
    selection = st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Chunk ID": st.column_config.NumberColumn(width="small"),
            "Images": st.column_config.NumberColumn(width="small"),
            "Tables": st.column_config.NumberColumn(width="small"),
            "Raw Len": st.column_config.NumberColumn(width="medium"),
            "Enh Len": st.column_config.NumberColumn(width="medium"),
            "Health": st.column_config.ProgressColumn(
                format="$score / 100",
                min_value=0,
                max_value=100,
                width="small",
            ),
        },
        on_select="rerun",
        selection_mode="single-row",
    )
    if selection and selection.selection.rows:
        idx = selection.selection.rows[0]
        st.session_state.selected_chunk_id = data[idx].get("chunk_id")


def _health_color(score: int) -> str:
    if score >= 75:
        return "green"
    elif score >= 50:
        return "orange"
    return "red"


def _status_badge(present: bool, label: str) -> str:
    if present:
        return f":green[PASS] {label}"
    return f":red[FAIL] {label}"


def render_detail(chunk: dict) -> None:
    cid = chunk.get("chunk_id", "?")
    images = get_images_base64(chunk)
    tables = get_tables_html(chunk)
    raw = get_raw_text(chunk)
    enhanced = get_enhanced_content(chunk)

    score, label = health_score(chunk)
    color = _health_color(score)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Chunk ID", cid)
    col2.metric("Images", len(images))
    col3.metric("Tables", len(tables))
    col4.metric("Raw Chars", len(raw))
    col5.metric("Enhanced Chars", len(enhanced))

    st.markdown(f"### Health Score: {score}/100  --  :{color}[{label}]")
    st.progress(score / 100)

    qc1, qc2, qc3, qc4 = st.columns(4)
    qc1.markdown(_status_badge(bool(enhanced), "Enhanced Content"))
    qc2.markdown(_status_badge(bool(raw), "Raw Text"))
    qc3.markdown(_status_badge(bool(images), "Images"))
    qc4.markdown(_status_badge(bool(tables), "Tables"))

    col_e1, col_e2, _ = st.columns([1, 1, 6])
    with col_e1:
        st.download_button(
            "Export JSON",
            data=json.dumps(chunk, indent=2),
            file_name=f"chunk_{cid}.json",
            mime="application/json",
        )
    with col_e2:
        export_txt = (
            f"Chunk {cid}\n\n"
            f"--- Enhanced ---\n{enhanced}\n\n"
            f"--- Raw ---\n{raw}"
        )
        st.download_button(
            "Export TXT",
            data=export_txt,
            file_name=f"chunk_{cid}.txt",
            mime="text/plain",
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Enhanced Content", "Raw Text", "Images", "Tables", "Metadata"]
    )

    with tab1:
        _render_enhanced_tab(enhanced)
    with tab2:
        _render_raw_tab(raw)
    with tab3:
        _render_images_tab(images)
    with tab4:
        _render_tables_tab(tables)
    with tab5:
        _render_metadata_tab(chunk)


def _render_enhanced_tab(text: str) -> None:
    st.markdown(text)

    wc = word_count(text)
    cc = len(text)

    col1, col2 = st.columns(2)
    col1.metric("Words", wc)
    col2.metric("Characters", cc)

    with st.expander("Show / Hide"):
        st.markdown(text)


def _render_raw_tab(text: str) -> None:
    wc = word_count(text)
    cc = len(text)

    col1, col2 = st.columns(2)
    col1.metric("Words", wc)
    col2.metric("Characters", cc)

    st.text_area(
        "Raw Text",
        value=text,
        height=400,
        disabled=True,
        label_visibility="collapsed",
    )

    st.download_button(
        "Download", data=text, file_name="raw_text.txt", mime="text/plain"
    )


def _render_images_tab(images: list[str]) -> None:
    if not images:
        st.info("No images found in this chunk.")
        return

    for i, b64 in enumerate(images, 1):
        st.subheader(f"Image {i}")
        img = render_image_from_base64(b64)
        if img is None:
            st.warning(f"Image {i} could not be decoded (malformed base64).")
            continue

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        bts = buf.getvalue()

        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(img, width="stretch")
        with col2:
            st.write(f"**Width:** {img.width} px")
            st.write(f"**Height:** {img.height} px")
            st.write(f"**Size:** {len(bts) / 1024:.1f} KB")
            st.write(f"**Format:** {img.format or 'N/A'}")

            st.download_button(
                "Download",
                data=bts,
                file_name=f"chunk_image_{i}.png",
                mime="image/png",
                key=f"dl_img_{i}",
            )

        with st.expander(f"Zoom Image {i}"):
            st.image(img, width=800)


def _render_tables_tab(tables: list[str]) -> None:
    if not tables:
        st.info("No tables found in this chunk.")
        return

    for i, html in enumerate(tables, 1):
        st.subheader(f"Table {i}")
        df = parse_table_html(html)
        if df is not None:
            st.dataframe(df, width="stretch")
        else:
            st.warning(
                f"Could not parse Table {i} as DataFrame. Showing raw HTML."
            )
            st.html(html)

        with st.expander(f"Raw HTML -- Table {i}"):
            st.code(html, language="html")


def _render_metadata_tab(chunk: dict) -> None:
    meta = chunk.get("metadata", {})
    images = get_images_base64(chunk)
    tables = get_tables_html(chunk)

    col1, col2, col3 = st.columns(3)
    col1.metric("Image Count", len(images))
    col2.metric("Table Count", len(tables))
    col3.write(f"**Metadata keys:** {list(meta.keys())}")

    st.divider()
    st.json(meta)


def render_compare(data: list[dict]) -> None:
    st.header("Compare Chunks")

    ids = [c.get("chunk_id") for c in data]

    col_a, col_b = st.columns(2)
    with col_a:
        a_id = st.selectbox("Chunk A", ids, key="cmp_a")
    with col_b:
        b_id = st.selectbox("Chunk B", ids, key="cmp_b")

    chunk_a = next((c for c in data if c.get("chunk_id") == a_id), None)
    chunk_b = next((c for c in data if c.get("chunk_id") == b_id), None)

    if not chunk_a or not chunk_b:
        st.warning("Select two chunks to compare.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Chunk {a_id}")
        _render_mini_chunk(chunk_a)
    with col2:
        st.subheader(f"Chunk {b_id}")
        _render_mini_chunk(chunk_b)


def _render_mini_chunk(chunk: dict) -> None:
    images = get_images_base64(chunk)
    tables = get_tables_html(chunk)
    raw = get_raw_text(chunk)
    enhanced = get_enhanced_content(chunk)
    score, label = health_score(chunk)
    color = _health_color(score)

    st.metric("Health Score", f"{score}/100")
    st.progress(score / 100)
    st.write(f"**Images:** {len(images)}  |  **Tables:** {len(tables)}")
    st.write(
        f"**Raw:** {len(raw)} chars  |  **Enhanced:** {len(enhanced)} chars"
    )

    with st.expander("Enhanced Content"):
        st.markdown(enhanced or "*empty*")
    with st.expander("Raw Text"):
        st.text(raw or "*empty*")
    with st.expander(f"Images ({len(images)})"):
        for i, b64 in enumerate(images, 1):
            img = render_image_from_base64(b64)
            if img:
                st.image(img, width=300, caption=f"Image {i}")
    with st.expander(f"Tables ({len(tables)})"):
        for i, html in enumerate(tables, 1):
            df = parse_table_html(html)
            if df is not None:
                st.dataframe(df)
            else:
                st.html(html)
    with st.expander("Metadata"):
        st.json(chunk.get("metadata", {}))


def render_analytics(data: list[dict]) -> None:
    st.header("Dataset Analytics")

    df = pd.DataFrame(
        [
            {
                "chunk_id": c.get("chunk_id"),
                "images": len(get_images_base64(c)),
                "tables": len(get_tables_html(c)),
                "raw_len": len(get_raw_text(c)),
                "enhanced_len": len(get_enhanced_content(c)),
            }
            for c in data
        ]
    )

    if df.empty:
        st.info("No data to analyse.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Images", df["images"].sum())
    m2.metric("Total Tables", df["tables"].sum())
    m3.metric("Avg Chunk Size (chars)", f"{df['raw_len'].mean():.0f}")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(
            df,
            x="images",
            title="Distribution of Images per Chunk",
            labels={"images": "Image Count", "count": "Chunks"},
            nbins=df["images"].max() + 1,
        )
        st.plotly_chart(fig1, width="stretch")
    with col2:
        fig2 = px.histogram(
            df,
            x="tables",
            title="Distribution of Tables per Chunk",
            labels={"tables": "Table Count", "count": "Chunks"},
            nbins=df["tables"].max() + 1,
        )
        st.plotly_chart(fig2, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.histogram(
            df,
            x="raw_len",
            title="Distribution of Raw Text Length",
            labels={"raw_len": "Character Count", "count": "Chunks"},
        )
        st.plotly_chart(fig3, width="stretch")
    with col4:
        fig4 = px.histogram(
            df,
            x="enhanced_len",
            title="Distribution of Enhanced Content Length",
            labels={"enhanced_len": "Character Count", "count": "Chunks"},
        )
        st.plotly_chart(fig4, width="stretch")

    st.subheader("Top 20 Lists")
    tcol1, tcol2, tcol3 = st.columns(3)

    with tcol1:
        st.markdown("**Largest Chunks (raw text)**")
        top_raw = df.nlargest(20, "raw_len")[["chunk_id", "raw_len"]]
        st.dataframe(top_raw, width="stretch")

    with tcol2:
        st.markdown("**Most Images**")
        top_img = df.nlargest(20, "images")[["chunk_id", "images"]]
        st.dataframe(top_img, width="stretch")

    with tcol3:
        st.markdown("**Most Tables**")
        top_tbl = df.nlargest(20, "tables")[["chunk_id", "tables"]]
        st.dataframe(top_tbl, width="stretch")


def page_explorer(data: list[dict]) -> None:
    render_dashboard(data)

    filtered = apply_filters(data)
    st.info(f"Showing {len(filtered)} of {len(data)} chunks")

    if not filtered:
        st.warning("No chunks match the current filters.")
        return

    list_col, detail_col = st.columns([1, 1.8])

    with list_col:
        render_chunk_list(filtered)
        cid_filter = st.session_state.chunk_id_filter.strip()
        if not cid_filter:
            sel = st.selectbox(
                "Select chunk",
                options=[c.get("chunk_id") for c in filtered],
                format_func=lambda x: f"Chunk {x}",
                key="chunk_selector",
                label_visibility="collapsed",
                index=None,
                placeholder="Pick a chunk...",
            )
            if sel is not None:
                st.session_state.selected_chunk_id = sel
                st.rerun()

    with detail_col:
        selected_id = st.session_state.selected_chunk_id
        if selected_id is not None:
            chunk = next(
                (c for c in data if c.get("chunk_id") == selected_id), None
            )
            if chunk:
                with st.container(border=True):
                    render_detail(chunk)
            else:
                st.info("Select a chunk from the list.")


# ---------------------------------------------------------------------------
# Query & Retrieve page (talks to the FastAPI backend)
# ---------------------------------------------------------------------------
def _render_chunk_payload(payload: dict, index: int) -> None:
    """Render one retrieved chunk from an API ``ChunkPayload`` dict."""
    raw_text = payload.get("raw_text", "")
    tables = payload.get("tables_html", [])
    image_urls = payload.get("image_urls", [])
    enhanced = payload.get("enhanced_content", "")

    label = (
        f"Chunk {index + 1}  —  "
        f"Images: {len(image_urls)}  |  "
        f"Tables: {len(tables)}  |  "
        f"Raw: {len(raw_text)} chars  |  "
        f"Enhanced: {len(enhanced)} chars"
    )

    with st.expander(label, expanded=index == 0):
        tab_e, tab_r, tab_i, tab_t = st.tabs(["Enhanced", "Raw Text", "Images", "Tables"])

        with tab_e:
            st.markdown(enhanced)

        with tab_r:
            st.metric("Words", len(raw_text.split()))
            st.text_area(
                "raw",
                value=raw_text,
                height=250,
                disabled=True,
                label_visibility="collapsed",
                key=f"api_raw_{index}",
            )

        with tab_i:
            if not image_urls:
                st.info("No images in this chunk.")
            else:
                for j, url in enumerate(image_urls):
                    st.image(url, width=400, caption=f"Image {j + 1}")

        with tab_t:
            if not tables:
                st.info("No tables in this chunk.")
            else:
                for j, html in enumerate(tables):
                    st.markdown(f"**Table {j + 1}**")
                    df = parse_table_html(html)
                    if df is not None:
                        st.dataframe(df, width="stretch")
                    else:
                        st.warning("Could not parse table.")
                        st.code(html, language="html")


def _render_retrieval(retrieval: dict) -> None:
    """Render the ``chunks`` of a retrieve/query API response."""
    chunks = retrieval.get("chunks", [])
    with st.expander(f"**Referenced Chunks** — {len(chunks)} chunks", expanded=True):
        for i, payload in enumerate(chunks):
            _render_chunk_payload(payload, i)


def render_chat_page() -> None:
    st.header("Query & Retrieve")

    st.markdown(
        "Enter a query to search the vector store and generate "
        "a multimodal answer with referenced chunks. All retrieval and "
        "generation runs on the FastAPI backend."
    )

    # Check the backend is reachable before doing anything else.
    try:
        backend_status = api_health()
        backend_ok = backend_status.get("status") == "ok"
    except Exception:
        backend_ok = False

    if not backend_ok:
        st.error(
            f"Backend API unreachable at `{API_BASE_URL}`. "
            f"Start it with `uvicorn api.main:app --reload --port 8000`."
        )
    else:
        st.success(f"Backend connected — vector store: {backend_status.get('vector_store')}")

    query = st.text_area(
        "Query",
        value=st.session_state.chat_query,
        placeholder="Ask a question about your documents ...",
        key="chat_query_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submitted = st.button("Search", type="primary", width="stretch")
    with col2:
        clear = st.button("Clear", width="stretch")
        if clear:
            st.session_state.chat_answer = None
            st.session_state.chat_retrieval = None
            st.rerun()

    if submitted and query.strip():
        st.session_state.chat_query = query
        try:
            events = api_stream(query)
            status = st.status("Retrieving relevant chunks ...", expanded=True)
            answer_placeholder = st.empty()
            collected: list[str] = []

            for event, data in events:
                if event == "retrieval":
                    retrieval = data
                    chunks = retrieval.get("chunks", [])
                    st.session_state.chat_retrieval = retrieval
                    status.update(
                        label=f"Retrieved {len(chunks)} chunks",
                        state="complete",
                        expanded=False,
                    )
                    _render_retrieval(retrieval)
                elif event == "token":
                    collected.append(data.get("delta", ""))
                    answer_placeholder.markdown(f"### Answer\n\n{''.join(collected)}▌")
                elif event == "done":
                    answer_placeholder.markdown(f"### Answer\n\n{data.get('answer', '')}")
                    st.session_state.chat_answer = data.get("answer", "")
                elif event == "error":
                    status.update(label="Answer generation failed", state="error")
                    st.error(data.get("message", "Unknown backend error"))

            if not collected:
                answer_placeholder.markdown("### Answer\n\n*No answer returned.*")
        except Exception as e:
            st.error(f"Query failed: {e}")

    # Display persisted results on subsequent script runs.
    answer = st.session_state.get("chat_answer")
    retrieval = st.session_state.get("chat_retrieval")

    if answer is None and retrieval is None:
        st.info("Enter a query and press Search to get started.")
        return

    if submitted and query.strip():
        return

    if retrieval:
        _render_retrieval(retrieval)

    if answer:
        st.markdown("### Answer")
        st.markdown(answer)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
        div[data-testid="stMetricLabel"] { font-size: 0.75rem; }
        div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th {
            font-size: 0.8rem;
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        hr { margin: 0.5rem 0; }
        h1 { font-size: 1.6rem; margin-bottom: 0; }
        h2 { font-size: 1.2rem; }
        h3 { font-size: 1.0rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(PAGE_TITLE)
    init_state()

    page = st.sidebar.radio(
        "Navigation",
        [
            "Chunk Explorer",
            "Compare Chunks",
            "Dataset Analytics",
            "Query & Retrieve",
            "Parallel Model Test",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    # The JSON-chunk data loader belongs to Chunk Explorer / Compare / Analytics.
    # Parallel Model Test takes datasheets as PDFs (upload or URL) directly in
    # its own "Run Test" tab, so the JSON uploader would just be confusing there.
    render_sidebar(show_data_section=page != "Parallel Model Test")

    data = st.session_state.data

    if st.session_state.file_name:
        st.sidebar.caption(
            f"Loaded: {st.session_state.file_name}  "
            f"-  Chunks: {len(data)}"
        )

    if page == "Query & Retrieve":
        render_chat_page()
    elif page == "Parallel Model Test":
        model_test_page.render()
    elif data is None:
        st.info("Upload a JSON file using the sidebar to get started.")
    elif page == "Chunk Explorer":
        page_explorer(data)
    elif page == "Compare Chunks":
        render_compare(data)
    elif page == "Dataset Analytics":
        render_analytics(data)


if __name__ == "__main__":
    log_tracing_status()
    main()
