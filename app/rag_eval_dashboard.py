"""RAG Evaluation Dashboard — Streamlit app for visualising RAG eval results."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    h1 { font-size: 1.6rem; margin-bottom: 0; }
    h2 { font-size: 1.2rem; }
    h3 { font-size: 1.0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "evals" / "testing.json"

GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"]

RETRIEVAL_METRICS = ["relevance", "coverage", "ranking", "redundancy", "noise"]
GENERATION_METRICS = ["correctness", "faithfulness", "completeness", "conciseness"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalise_retrieved_docs(raw) -> list[str]:
    """Return a flat list of document names regardless of input format."""
    if isinstance(raw, list):
        return [str(d) for d in raw]
    if isinstance(raw, dict):
        return [f"{k} (x{v})" for k, v in raw.items()]
    return []


def extract_doc_names(raw) -> list[str]:
    """Return clean document names (without count suffixes)."""
    if isinstance(raw, list):
        return [str(d) for d in raw]
    if isinstance(raw, dict):
        return list(raw.keys())
    return []


def has_contamination(raw) -> bool:
    """True when more than one distinct source document appears."""
    return len(set(extract_doc_names(raw))) > 1


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

def render_sidebar(data: list[dict]) -> list[dict]:
    result = list(data)

    with st.sidebar:
        st.header("Filters")

        # Grade filter
        all_grades = sorted(
            set(r["overall"]["grade"] for r in result),
            key=lambda g: GRADE_ORDER.index(g) if g in GRADE_ORDER else 99,
        )
        selected_grades = st.multiselect("Grade", all_grades, default=all_grades)

        # Status filter
        statuses = sorted(set(r["overall"]["status"] for r in result))
        selected_statuses = st.multiselect("Status", statuses, default=statuses)

        # Min score slider
        scores = [r["overall"]["score"] for r in result]
        min_score, max_score = min(scores), max(scores)
        score_range = st.slider(
            "Overall score range",
            min_value=min_score,
            max_value=max_score,
            value=(min_score, max_score),
        )

        # Search box
        search = st.text_input("Search questions", placeholder="Type to filter...")

        st.divider()
        st.caption(f"Showing records after filters applied")

    # Apply filters
    filtered = [
        r
        for r in result
        if r["overall"]["grade"] in selected_grades
        and r["overall"]["status"] in selected_statuses
        and score_range[0] <= r["overall"]["score"] <= score_range[1]
    ]

    if search.strip():
        q = search.strip().lower()
        filtered = [r for r in filtered if q in r["question"].lower()]

    return filtered


# ---------------------------------------------------------------------------
# Summary cards
# ---------------------------------------------------------------------------

def render_summary(data: list[dict]) -> None:
    if not data:
        st.warning("No records match the current filters.")
        return

    n = len(data)
    avg_overall = sum(r["overall"]["score"] for r in data) / n
    pass_rate = sum(1 for r in data if r["overall"]["status"] == "Pass") / n * 100
    avg_retrieval = sum(r["retrieval"]["retrieval_score"] for r in data) / n
    avg_generation = sum(r["generation"]["generation_score"] for r in data) / n

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", n)
    c2.metric("Avg Overall Score", f"{avg_overall:.1f}")
    c3.metric("Pass Rate", f"{pass_rate:.0f}%")
    c4.metric("Avg Retrieval Score", f"{avg_retrieval:.1f}")
    c5.metric("Avg Generation Score", f"{avg_generation:.1f}")


# ---------------------------------------------------------------------------
# Score distribution charts
# ---------------------------------------------------------------------------

def render_score_charts(data: list[dict]) -> None:
    if not data:
        return

    st.subheader("Score Distribution")

    # --- Bar chart: overall score per question ---
    df_scores = pd.DataFrame(
        [
            {
                "id": r["id"],
                "score": r["overall"]["score"],
                "grade": r["overall"]["grade"],
                "status": r["overall"]["status"],
                "question_short": r["question"][:60] + ("..." if len(r["question"]) > 60 else ""),
            }
            for r in data
        ]
    )
    df_scores["label"] = df_scores.apply(
        lambda row: f"Q{row['id']} [{row['grade']}]", axis=1
    )
    fig_bar = px.bar(
        df_scores,
        x="label",
        y="score",
        color="status",
        color_discrete_map={"Pass": "#2ecc71", "Fail": "#e74c3c"},
        hover_data=["question_short", "grade"],
        title="Overall Score per Question",
        labels={"label": "Question", "score": "Overall Score"},
    )
    fig_bar.update_layout(yaxis_range=[0, 105], xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Grouped bar: retrieval vs generation ---
    df_compare = pd.DataFrame(
        [
            {
                "id": r["id"],
                "retrieval_score": r["retrieval"]["retrieval_score"],
                "generation_score": r["generation"]["generation_score"],
                "label": f"Q{r['id']}",
            }
            for r in data
        ]
    )
    fig_grouped = go.Figure()
    fig_grouped.add_trace(
        go.Bar(
            x=df_compare["label"],
            y=df_compare["retrieval_score"],
            name="Retrieval",
            marker_color="#3498db",
        )
    )
    fig_grouped.add_trace(
        go.Bar(
            x=df_compare["label"],
            y=df_compare["generation_score"],
            name="Generation",
            marker_color="#e67e22",
        )
    )
    fig_grouped.update_layout(
        barmode="group",
        title="Retrieval vs Generation Score per Question",
        xaxis_title="Question",
        yaxis_title="Score",
        yaxis_range=[0, 105],
    )
    st.plotly_chart(fig_grouped, use_container_width=True)

    # --- Histogram of grade distribution ---
    grade_counts = Counter(r["overall"]["grade"] for r in data)
    df_grades = pd.DataFrame(
        [
            {"grade": g, "count": grade_counts.get(g, 0)}
            for g in GRADE_ORDER
            if g in grade_counts
        ]
    )
    fig_hist = px.bar(
        df_grades,
        x="grade",
        y="count",
        color="grade",
        title="Grade Distribution",
        labels={"grade": "Grade", "count": "Count"},
    )
    fig_hist.update_layout(
        xaxis={"categoryorder": "array", "categoryarray": GRADE_ORDER},
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ---------------------------------------------------------------------------
# Retrieval metrics breakdown
# ---------------------------------------------------------------------------

def render_retrieval_metrics(data: list[dict]) -> None:
    if not data:
        return

    st.subheader("Retrieval Metrics Breakdown")

    # Average metrics
    avg_metrics = {}
    for m in RETRIEVAL_METRICS:
        avg_metrics[m] = sum(r["retrieval"]["metrics"][m] for r in data) / len(data)

    # --- Radar chart ---
    categories = RETRIEVAL_METRICS + [RETRIEVAL_METRICS[0]]
    values = [avg_metrics[m] for m in RETRIEVAL_METRICS] + [avg_metrics[RETRIEVAL_METRICS[0]]]

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Avg Metrics",
            line_color="#3498db",
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        title="Average Retrieval Metrics (1-5 scale)",
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Also show as grouped bar for clarity
    df_ret = pd.DataFrame(
        [{"metric": m.capitalize(), "score": avg_metrics[m]} for m in RETRIEVAL_METRICS]
    )
    fig_ret_bar = px.bar(
        df_ret,
        x="metric",
        y="score",
        color="metric",
        title="Average Retrieval Sub-Metrics",
        labels={"score": "Score (1-5)"},
    )
    fig_ret_bar.update_layout(showlegend=False, yaxis_range=[0, 5.5])
    st.plotly_chart(fig_ret_bar, use_container_width=True)

    # --- Documents table ---
    st.markdown("**Retrieved Documents per Question**")
    doc_rows = []
    for r in data:
        raw_docs = r["retrieval"]["retrieved_documents"]
        doc_names = extract_doc_names(raw_docs)
        contam = has_contamination(raw_docs)
        doc_rows.append(
            {
                "ID": r["id"],
                "Question": r["question"][:80] + ("..." if len(r["question"]) > 80 else ""),
                "Retrieved Documents": ", ".join(doc_names),
                "Distinct Sources": len(set(doc_names)),
                "Contamination": "Yes" if contam else "No",
            }
        )
    df_docs = pd.DataFrame(doc_rows)
    st.dataframe(
        df_docs,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Generation metrics breakdown
# ---------------------------------------------------------------------------

def render_generation_metrics(data: list[dict]) -> None:
    if not data:
        return

    st.subheader("Generation Metrics Breakdown")

    # Average metrics (excluding image_utilization)
    avg_metrics = {}
    for m in GENERATION_METRICS:
        avg_metrics[m] = sum(r["generation"]["metrics"][m] for r in data) / len(data)

    # --- Radar chart ---
    categories = GENERATION_METRICS + [GENERATION_METRICS[0]]
    values = [avg_metrics[m] for m in GENERATION_METRICS] + [avg_metrics[GENERATION_METRICS[0]]]

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Avg Metrics",
            line_color="#e67e22",
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        title="Average Generation Metrics (1-5 scale)",
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Grouped bar
    df_gen = pd.DataFrame(
        [{"metric": m.capitalize(), "score": avg_metrics[m]} for m in GENERATION_METRICS]
    )
    fig_gen_bar = px.bar(
        df_gen,
        x="metric",
        y="score",
        color="metric",
        title="Average Generation Sub-Metrics",
        labels={"score": "Score (1-5)"},
    )
    fig_gen_bar.update_layout(showlegend=False, yaxis_range=[0, 5.5])
    st.plotly_chart(fig_gen_bar, use_container_width=True)

    # --- Image Utilization ---
    st.markdown("**Image Utilization**")
    img_vals = [r["generation"]["metrics"]["image_utilization"] for r in data]
    na_count = sum(1 for v in img_vals if v is None or v == "NA" or v == "N/A")
    numeric_vals = [v for v in img_vals if isinstance(v, (int, float))]
    avg_img = sum(numeric_vals) / len(numeric_vals) if numeric_vals else 0

    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("Total Records", len(data))
    ic2.metric("N/A Count", na_count)
    ic3.metric("Avg (numeric only)", f"{avg_img:.2f}")

    if numeric_vals:
        df_img = pd.DataFrame(
            [
                {
                    "ID": r["id"],
                    "image_utilization": r["generation"]["metrics"]["image_utilization"],
                }
                for r in data
                if isinstance(r["generation"]["metrics"]["image_utilization"], (int, float))
            ]
        )
        fig_img = px.bar(
            df_img,
            x="ID",
            y="image_utilization",
            title="Image Utilization per Question (numeric values only)",
            labels={"image_utilization": "Score"},
        )
        fig_img.update_layout(yaxis_range=[0, 5.5])
        st.plotly_chart(fig_img, use_container_width=True)


# ---------------------------------------------------------------------------
# Failure / weak-spot analysis
# ---------------------------------------------------------------------------

def render_failure_analysis(data: list[dict]) -> None:
    if not data:
        return

    st.subheader("Failure & Weak-Spot Analysis")

    failures = [
        r for r in data
        if r["overall"]["status"] == "Fail" or r["overall"]["score"] < 80
    ]

    if not failures:
        st.success("No failures or weak spots detected (all scores >= 80 and status Pass).")
        return

    # Failure table
    fail_rows = []
    for r in failures:
        diag = r["diagnosis"]
        fail_rows.append(
            {
                "ID": r["id"],
                "Question": r["question"][:80] + ("..." if len(r["question"]) > 80 else ""),
                "Score": r["overall"]["score"],
                "Grade": r["overall"]["grade"],
                "Status": r["overall"]["status"],
                "Root Cause": diag["root_cause"],
                "Weaknesses": "; ".join(diag["weaknesses"]),
                "Suggested Fix": diag["suggested_fix"],
            }
        )
    df_fail = pd.DataFrame(fail_rows)
    st.dataframe(df_fail, use_container_width=True, hide_index=True)

    # Root cause theme analysis
    st.markdown("**Root Cause Themes**")
    all_causes = " ".join(r["diagnosis"]["root_cause"] for r in data)
    # Simple keyword extraction
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
        "either", "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "because", "if", "when", "where", "how",
    }
    words = re.findall(r"[a-zA-Z]+", all_causes.lower())
    meaningful = [w for w in words if w not in stop_words and len(w) > 2]
    word_freq = Counter(meaningful).most_common(15)

    if word_freq:
        df_wc = pd.DataFrame(word_freq, columns=["Theme", "Count"])
        fig_wc = px.bar(
            df_wc,
            x="Theme",
            y="Count",
            title="Recurring Root Cause Themes",
        )
        st.plotly_chart(fig_wc, use_container_width=True)


# ---------------------------------------------------------------------------
# Detail drill-down
# ---------------------------------------------------------------------------

def render_detail_drilldown(data: list[dict]) -> None:
    if not data:
        return

    st.subheader("Detail Drill-Down")

    ids = [r["id"] for r in data]
    selected_id = st.selectbox(
        "Select a question by ID",
        options=ids,
        format_func=lambda x: f"Q{x} — {next(r['question'][:60] for r in data if r['id'] == x)}...",
    )

    record = next(r for r in data if r["id"] == selected_id)

    # --- Colored badge ---
    status = record["overall"]["status"]
    grade = record["overall"]["grade"]
    score = record["overall"]["score"]
    badge_color = "green" if status == "Pass" else "red"
    st.markdown(
        f"### :{badge_color}[{status}] — Grade: {grade} — Score: {score}/100"
    )

    # --- Question & Answer ---
    st.markdown(f"**Question:** {record['question']}")
    st.markdown(f"**Model Answer:** {record['generation']['model_answer']}")
    st.markdown(f"**Answer Summary:** {record['generation']['answer_summary']}")

    # --- Retrieved Documents ---
    raw_docs = record["retrieval"]["retrieved_documents"]
    st.markdown(f"**Retrieved Documents:** {normalise_retrieved_docs(raw_docs)}")
    st.markdown(f"**Retrieval Summary:** {record['retrieval']['retrieved_summary']}")

    # --- Metrics tables ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Retrieval Metrics**")
        ret_metrics = record["retrieval"]["metrics"]
        ret_df = pd.DataFrame(
            [{"Metric": k.capitalize(), "Score": v} for k, v in ret_metrics.items()]
        )
        st.dataframe(ret_df, hide_index=True, use_container_width=True)
        st.metric("Retrieval Score", f"{record['retrieval']['retrieval_score']}/100")

    with col2:
        st.markdown("**Generation Metrics**")
        gen_metrics = record["generation"]["metrics"]
        gen_rows = []
        for k, v in gen_metrics.items():
            display_val = "N/A" if v is None or v == "NA" else v
            gen_rows.append({"Metric": k.replace("_", " ").capitalize(), "Score": display_val})
        gen_df = pd.DataFrame(gen_rows)
        st.dataframe(gen_df, hide_index=True, use_container_width=True)
        st.metric("Generation Score", f"{record['generation']['generation_score']}/100")

    # --- Strengths & Weaknesses ---
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Strengths**")
        for s in record["diagnosis"]["strengths"]:
            st.markdown(f"- {s}")
    with col4:
        st.markdown("**Weaknesses**")
        for w in record["diagnosis"]["weaknesses"]:
            st.markdown(f"- {w}")

    # --- Diagnosis ---
    st.markdown(f"**Root Cause:** {record['diagnosis']['root_cause']}")
    st.markdown(f"**Suggested Fix:** {record['diagnosis']['suggested_fix']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("RAG Evaluation Dashboard")

    # Load data
    try:
        data = load_data(str(DATA_PATH))
    except FileNotFoundError:
        st.error(f"Data file not found at `{DATA_PATH}`.")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON: {e}")
        st.stop()

    # Apply sidebar filters
    filtered = render_sidebar(data)

    # Summary
    render_summary(filtered)

    if not filtered:
        st.stop()

    st.divider()

    # Tabs for main sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Score Distribution",
            "Retrieval Metrics",
            "Generation Metrics",
            "Failure Analysis",
            "Detail Drill-Down",
        ]
    )

    with tab1:
        render_score_charts(filtered)

    with tab2:
        render_retrieval_metrics(filtered)

    with tab3:
        render_generation_metrics(filtered)

    with tab4:
        render_failure_analysis(filtered)

    with tab5:
        render_detail_drilldown(filtered)


if __name__ == "__main__":
    main()
