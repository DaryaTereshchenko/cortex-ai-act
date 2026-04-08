"""Web-UI — Streamlit entry point for CORTEX-RAG."""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="CORTEX-RAG | Regulatory Compliance Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# API base URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

# Page styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .workspace-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #0F4C81;
        text-align: center;
        margin-top: 0.25rem;
        letter-spacing: 0.02em;
    }

    .workspace-subtitle {
        font-size: 0.98rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.35rem;
    }

    .panel-tag {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6B7280;
        margin-bottom: 0.25rem;
    }

    .panel-shell {
        background: linear-gradient(180deg, rgba(250,251,255,0.95) 0%, rgba(246,248,252,0.95) 100%);
        border: 1px solid #D6E0EA;
        border-radius: 12px;
        padding: 0.75rem;
        margin-bottom: 1rem;
    }

    .answer-box {
        background-color: #EEF6FF;
        border-left: 4px solid #0F4C81;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0 1rem 0;
        color: #0B1F33;
        line-height: 1.55;
    }

    .citation-box {
        background-color: #FFF7E6;
        border-left: 4px solid #D97706;
        padding: 0.75rem;
        margin: 0.35rem 0;
        border-radius: 0.35rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #1F2937;
    }

    .about-card {
        border: 1px solid #D6E0EA;
        border-radius: 10px;
        padding: 0.9rem;
        background: #FCFDFF;
        margin-top: 0.25rem;
    }

    .info-dot {
        width: 2.1rem;
        height: 2.1rem;
        border-radius: 999px;
        border: 1px solid #BFD2E3;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #0F4C81;
        font-weight: 700;
        font-size: 1rem;
        background: #F6FAFF;
    }

    div[data-testid="stExpander"] > details > summary {
        font-weight: 600;
        color: #0F4C81;
    }

    .control-note {
        color: #6B7280;
        font-size: 0.86rem;
        margin-top: 0.25rem;
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F4C81;
        text-align: center;
        padding: 0.75rem 0 0.1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "current_query_id" not in st.session_state:
    st.session_state.current_query_id = None

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "query_input" not in st.session_state:
    st.session_state.query_input = ""


def get_system_health() -> dict:
    """Return service health status for drawer indicators."""
    status = {
        "api": False,
        "reasoning_engine_available": False,
        "knowledge_graph_available": False,
        "error": "",
    }
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        status["api"] = response.status_code == 200
        if response.status_code == 200:
            payload = response.json()
            status["reasoning_engine_available"] = bool(
                payload.get("reasoning_engine_available")
            )
            status["knowledge_graph_available"] = bool(
                payload.get("knowledge_graph_available")
            )
        else:
            status["error"] = f"API unavailable ({response.status_code})"
    except Exception as exc:
        status["error"] = str(exc)
    return status


def submit_query(question: str, payload: dict) -> dict | None:
    """Submit query and poll until completed/failed/timeout."""
    response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=12)
    if response.status_code != 200:
        st.error(f"API Error: {response.status_code}")
        return None

    query_id = response.json().get("query_id")
    if not query_id:
        st.error("Query submission failed: missing query id")
        return None

    st.session_state.current_query_id = query_id
    progress_bar = st.progress(0)
    status_text = st.empty()

    max_attempts = 120
    for attempt in range(max_attempts):
        result_response = requests.get(f"{API_BASE_URL}/query/{query_id}", timeout=12)
        result = result_response.json()

        if result.get("status") == "completed":
            progress_bar.progress(100)
            status_text.success("Query completed")
            return result

        if result.get("status") == "failed":
            st.error(f"Query failed: {result.get('error', 'Unknown error')}")
            return None

        progress = min((attempt / max_attempts) * 100, 92)
        progress_bar.progress(int(progress) / 100)
        status_text.info(
            f"Processing... step {len(result.get('reasoning_steps', []))}"
        )
        time.sleep(1)

    st.warning("Query timeout")
    return None


def render_browse_panel() -> None:
    """Browse legal corpus content in the left panel."""
    st.subheader("Browse Regulations")

    browse_section = st.selectbox(
        "Select Section",
        options=[
            "Regulations",
            "Articles",
            "Chapters",
            "Definitions",
            "Recitals",
            "Annexes",
        ],
        key="browse_section",
    )

    try:
        if browse_section == "Regulations":
            response = requests.get(f"{API_BASE_URL}/regulations", timeout=10)
            if response.status_code == 200:
                regs = response.json()
                for reg in regs:
                    st.write(f"**{reg.get('title', 'Unknown')}** (ID: {reg.get('id')})")
            else:
                st.error("Failed to load regulations")

        elif browse_section == "Articles":
            b1, b2 = st.columns(2)
            with b1:
                reg_select = st.selectbox(
                    "Select Regulation:", ["eu_ai_act", "dsa"], key="browse_art_reg"
                )
            with b2:
                article_num = st.number_input(
                    "Article Number:", min_value=1, max_value=200, key="browse_art_no"
                )

            if st.button("Load Article", key="browse_art_btn"):
                response = requests.get(
                    f"{API_BASE_URL}/article/{reg_select}/{int(article_num)}",
                    timeout=10,
                )
                if response.status_code == 200:
                    article = response.json()
                    st.write(f"**Title:** {article.get('title', 'N/A')}")
                    st.write(f"**Content:** {article.get('content', 'N/A')}")
                    if article.get("children"):
                        st.write("**Sub-sections:**")
                        for child in article["children"]:
                            st.write(f"- {child.get('title', child.get('id'))}")
                else:
                    st.error(f"Article not found (Status: {response.status_code})")

        elif browse_section == "Chapters":
            b1, b2 = st.columns(2)
            with b1:
                reg_select = st.selectbox(
                    "Select Regulation:", ["eu_ai_act", "dsa"], key="browse_ch_reg"
                )
            with b2:
                chapter_num = st.text_input("Chapter Number:", key="browse_ch_no")

            if st.button("Load Chapter", key="browse_ch_btn") and chapter_num:
                response = requests.get(
                    f"{API_BASE_URL}/chapter/{reg_select}/{chapter_num}", timeout=10
                )
                if response.status_code == 200:
                    chapter = response.json()
                    st.write(f"**Title:** {chapter.get('title', 'N/A')}")
                    if chapter.get("children"):
                        st.write("**Articles in Chapter:**")
                        for child in chapter["children"]:
                            st.write(
                                f"- Article {child.get('number')}: {child.get('title', 'N/A')}"
                            )
                else:
                    st.error(f"Chapter not found (Status: {response.status_code})")

        elif browse_section == "Definitions":
            reg_select = st.selectbox(
                "Select Regulation:", ["eu_ai_act", "dsa"], key="browse_def_reg"
            )

            response = requests.get(f"{API_BASE_URL}/definitions/{reg_select}", timeout=10)
            if response.status_code == 200:
                definitions = response.json()
                for defn in definitions[:20]:
                    st.write(f"**{defn.get('term', 'Unknown')}:** {defn.get('definition', 'N/A')}")
            else:
                st.error("Failed to load definitions")

        elif browse_section == "Recitals":
            reg_select = st.selectbox(
                "Select Regulation:", ["eu_ai_act", "dsa"], key="browse_rec_reg"
            )

            response = requests.get(f"{API_BASE_URL}/recitals/{reg_select}", timeout=10)
            if response.status_code == 200:
                recitals = response.json()
                for recital in recitals[:20]:
                    st.write(
                        f"**Recital {recital.get('number')}:** {recital.get('text', 'N/A')[:150]}..."
                    )
            else:
                st.error("Failed to load recitals")

        elif browse_section == "Annexes":
            reg_select = st.selectbox(
                "Select Regulation:", ["eu_ai_act", "dsa"], key="browse_ann_reg"
            )

            response = requests.get(f"{API_BASE_URL}/annexes/{reg_select}", timeout=10)
            if response.status_code == 200:
                annexes = response.json()
                for annex in annexes:
                    st.write(f"**Annex {annex.get('number')}:** {annex.get('title', 'N/A')}")
            else:
                st.error("Failed to load annexes")

    except Exception as exc:
        st.error(f"Error: {exc!s}")

# Header
top_left, _top_right = st.columns([1, 18])
with top_left:
    with st.popover("ⓘ", help="About CORTEX-RAG"):
        st.markdown("### About CORTEX-RAG")
        st.markdown(
            """
            <div class="about-card">
            <strong>CORTEX-RAG</strong> is a high-precision regulatory discovery system for the EU AI Act and DSA.
            <br/><br/>
            <strong>Core capabilities</strong>: graph-guided retrieval, semantic entropy pruning, and agentic self-correction.
            <br/><br/>
            <strong>Stack</strong>: Streamlit, FastAPI, Neo4j, and containerized deployment.
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="workspace-title">⚖️ CORTEX-RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="workspace-subtitle">Entropy-Driven Regulatory Discovery for EU AI Act & DSA</div>',
    unsafe_allow_html=True,
)

# Two-panel workspace (left: question/browse, right: answer/history)
left_panel, right_panel = st.columns([1, 1], gap="large")

with left_panel:
    st.markdown('<div class="panel-tag">Question Space</div>', unsafe_allow_html=True)
    with st.container(border=True):
        left_tabs = st.tabs(["🔍 Query", "📖 Browse"])

        with left_tabs[0]:
            st.subheader("Ask a Regulatory Compliance Question")

            with st.expander("💡 Example Questions"):
                st.markdown(
                    """
                - What are the obligations for providers of high-risk AI systems?
                - How does the AI Act's transparency requirement overlap with DSA?
                - What is the definition of 'systemic risk' in both regulations?
                - Which AI systems are explicitly prohibited under Article 5?
                - What are the conformity assessment procedures?
                """
                )

            user_question = st.text_area(
                "Your Question",
                height=180,
                key="query_input",
                placeholder="e.g., What obligations does the EU AI Act impose on deployers of high-risk AI systems?",
            )

            q1, q2 = st.columns(2)
            with q1:
                submit_button = st.button(
                    "🚀 Submit Query", type="primary", use_container_width=True
                )
            with q2:
                clear_button = st.button("🗑️ Clear", use_container_width=True)

            if clear_button:
                st.session_state.query_input = ""
                st.rerun()

            if submit_button:
                if not user_question.strip():
                    st.warning("Please enter a question before submitting.")
                else:
                    payload = {
                        "question": user_question.strip(),
                        "regulation": st.session_state.get("cfg_regulation", "eu_ai_act"),
                        "max_hops": st.session_state.get("cfg_max_hops", 3),
                        "enable_pruning": st.session_state.get("cfg_pruning", True),
                        "enable_self_correction": st.session_state.get("cfg_self_correct", True),
                    }
                    try:
                        with st.spinner("CORTEX is reasoning over the graph..."):
                            result = submit_query(user_question.strip(), payload)
                        if result:
                            st.session_state.latest_result = result
                            st.session_state.query_history.insert(0, result)
                            st.success("Answer ready in the right panel.")
                    except Exception as exc:
                        st.error(f"Request failed: {exc!s}")

        with left_tabs[1]:
            render_browse_panel()

with right_panel:
    st.markdown('<div class="panel-tag">Answer Space</div>', unsafe_allow_html=True)
    with st.container(border=True):
        right_tabs = st.tabs(["📄 Answer", "📜 History"])

        with right_tabs[0]:
            latest = st.session_state.latest_result
            if not latest:
                st.info("Submit a query on the left panel to view answer details here.")
            else:
                st.subheader("Answer")
                st.caption(f"Question: {latest.get('question', 'N/A')}")
                st.markdown(
                    f'<div class="answer-box">{latest.get("final_answer", "No answer generated")}</div>',
                    unsafe_allow_html=True,
                )

                if latest.get("citations"):
                    st.markdown("**Citations**")
                    for citation in latest["citations"]:
                        st.markdown(
                            f'<div class="citation-box">{citation}</div>',
                            unsafe_allow_html=True,
                        )

                metrics = latest.get("metrics", {})
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Reasoning Steps", len(latest.get("reasoning_steps", [])))
                with m2:
                    reduction = metrics.get("entropy_reduction", 0)
                    st.metric(
                        "Context Reduction",
                        f"{reduction:.1%}" if reduction else "N/A",
                    )
                with m3:
                    tokens = metrics.get("tokens_saved", 0)
                    st.metric("Tokens Saved", f"{tokens:,}" if tokens else "N/A")
                with m4:
                    latency = metrics.get("latency_seconds", 0)
                    st.metric("Latency", f"{latency:.2f}s" if latency else "N/A")

                with st.expander("▼ Reasoning Trace", expanded=False):
                    reasoning_steps = latest.get("reasoning_steps", [])
                    if not reasoning_steps:
                        st.caption("No reasoning trace available for this answer.")
                    for step in reasoning_steps:
                        with st.container(border=True):
                            r1, r2 = st.columns([3, 1])
                            with r1:
                                st.write(f"**Step {step['step_number']}: {step['agent']}**")
                                st.write(step.get("action", ""))
                            with r2:
                                if step.get("entropy_reduction"):
                                    st.metric("Reduction", f"{step['entropy_reduction']:.1%}")

        with right_tabs[1]:
            st.subheader("Query History")
            if not st.session_state.query_history:
                st.info("No queries yet. Submit a question on the left panel.")
            else:
                for query in st.session_state.query_history:
                    title = query.get("question", "Untitled query")[:70]
                    with st.expander(f"🕐 {title}..."):
                        st.write(f"**Status:** {query.get('status', 'N/A')}")
                        st.write(f"**Time:** {query.get('timestamp', 'N/A')}")
                        if query.get("final_answer"):
                            st.write(f"**Answer Preview:** {query['final_answer'][:180]}...")

# Bottom collapsible drawer for controls and health (default minimized)
st.divider()
with st.expander("▲ Open Controls Drawer", expanded=False):
    st.markdown(
        '<div class="control-note">This section is minimized by default. Expand when needed to tune query behavior.</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.subheader("⚙️ Query Controls")
        st.selectbox(
            "Regulation Scope",
            options=["eu_ai_act", "dsa", "both"],
            key="cfg_regulation",
            format_func=lambda x: {
                "eu_ai_act": "🇪🇺 EU AI Act Only",
                "dsa": "📱 DSA Only",
                "both": "🔗 Both Regulations",
            }[x],
        )
        st.slider(
            "Max Graph Traversal Hops",
            min_value=1,
            max_value=5,
            value=3,
            key="cfg_max_hops",
            help="Maximum depth for graph traversal in reasoning",
        )
        st.checkbox(
            "Enable Semantic Entropy Pruning",
            value=True,
            key="cfg_pruning",
            help="Reduce context by filtering low-information paths",
        )
        st.checkbox(
            "Enable Agentic Self-Correction",
            value=True,
            key="cfg_self_correct",
            help="Use critic agent to verify and re-retrieve context",
        )

    with c2:
        st.subheader("🏥 System Health")
        health = get_system_health()
        if health["api"]:
            st.success("✅ API Gateway")
        else:
            st.error("❌ API Gateway")

        if health["reasoning_engine_available"]:
            st.success("✅ Reasoning Engine")
        else:
            st.warning("⚠️ Reasoning Engine")

        if health["knowledge_graph_available"]:
            st.success("✅ Knowledge Graph")
        else:
            st.warning("⚠️ Knowledge Graph")

        if health["error"] and not health["api"]:
            st.caption(f"Health error: {health['error']}")
