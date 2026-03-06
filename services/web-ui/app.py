"""Web-UI — Streamlit entry point for CORTEX-RAG."""

from __future__ import annotations

import os
import time
from datetime import datetime

import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="CORTEX-RAG | Regulatory Compliance Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API base URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

# Page styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3B82F6;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #9CA3AF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: rgba(59, 130, 246, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    .answer-box {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: inherit;
    }
    .citation-box {
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #F59E0B;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        font-family: monospace;
        font-size: 0.85rem;
        color: inherit;
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

# Header
st.markdown('<div class="main-header">⚖️ CORTEX-RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Entropy-Driven Regulatory Discovery for EU AI Act & DSA</div>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    regulation = st.selectbox(
        "Regulation Scope",
        options=["eu_ai_act", "dsa", "both"],
        format_func=lambda x: {
            "eu_ai_act": "🇪🇺 EU AI Act Only",
            "dsa": "📱 DSA Only",
            "both": "🔗 Both Regulations",
        }[x],
    )

    max_hops = st.slider(
        "Max Graph Traversal Hops",
        min_value=1,
        max_value=5,
        value=3,
        help="Maximum depth for graph traversal in reasoning",
    )

    enable_pruning = st.checkbox(
        "Enable Semantic Entropy Pruning",
        value=True,
        help="Reduce context by filtering low-information paths",
    )

    enable_self_correction = st.checkbox(
        "Enable Agentic Self-Correction",
        value=True,
        help="Use critic agent to verify and re-retrieve context",
    )

    st.divider()

    # System health
    st.subheader("🏥 System Health")
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health = health_response.json()
            if health.get("reasoning_engine_available"):
                st.success("✅ Reasoning Engine")
            else:
                st.warning("⚠️ Reasoning Engine")

            if health.get("knowledge_graph_available"):
                st.success("✅ Knowledge Graph")
            else:
                st.warning("⚠️ Knowledge Graph")
        else:
            st.error("❌ API Unavailable")
    except Exception:
        st.error("❌ API Unreachable")

# Main tabs
query_tab, browse_tab, history_tab, about_tab = st.tabs(["🔍 Query", "📖 Browse", "📜 History", "ℹ️ About"])

with query_tab:
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
        height=100,
        placeholder="e.g., What obligations does the EU AI Act impose on deployers of high-risk AI systems?",
    )

    col1, col2 = st.columns(2)

    with col1:
        submit_button = st.button("🚀 Submit Query", type="primary", use_container_width=True)

    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    if clear_button:
        st.rerun()

    if submit_button and user_question:
        with st.spinner("🧠 CORTEX is thinking..."):
            payload = {
                "question": user_question,
                "regulation": regulation,
                "max_hops": max_hops,
                "enable_pruning": enable_pruning,
                "enable_self_correction": enable_self_correction,
            }

            try:
                # Submit query
                submit_response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=10)

                if submit_response.status_code == 200:
                    query_id = submit_response.json()["query_id"]
                    st.session_state.current_query_id = query_id

                    # Poll for results
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    max_attempts = 120  # 120 seconds timeout
                    for attempt in range(max_attempts):
                        result_response = requests.get(f"{API_BASE_URL}/query/{query_id}")
                        result = result_response.json()

                        if result["status"] == "completed":
                            progress_bar.progress(100)
                            status_text.success("✅ Query completed!")
                            st.session_state.query_history.insert(0, result)

                            # Display results
                            st.divider()

                            # Answer
                            st.subheader("📄 Answer")
                            st.markdown(
                                f'<div class="answer-box">{result.get("final_answer", "No answer generated")}</div>',
                                unsafe_allow_html=True,
                            )

                            # Citations
                            if result.get("citations"):
                                st.subheader("📚 Citations")
                                for citation in result["citations"]:
                                    st.markdown(
                                        f'<div class="citation-box">{citation}</div>',
                                        unsafe_allow_html=True,
                                    )

                            # Metrics
                            metrics = result.get("metrics", {})
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric(
                                    "Reasoning Steps", len(result.get("reasoning_steps", []))
                                )

                            with col2:
                                reduction = metrics.get("entropy_reduction", 0)
                                st.metric(
                                    "Context Reduction",
                                    f"{reduction:.1%}" if reduction else "N/A",
                                )

                            with col3:
                                tokens = metrics.get("tokens_saved", 0)
                                st.metric("Tokens Saved", f"{tokens:,}" if tokens else "N/A")

                            with col4:
                                latency = metrics.get("latency_seconds", 0)
                                st.metric("Latency", f"{latency:.2f}s" if latency else "N/A")

                            # Reasoning trace
                            if result.get("reasoning_steps"):
                                st.subheader("🔄 Reasoning Trace")
                                for step in result["reasoning_steps"]:
                                    with st.container(border=True):
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            st.write(
                                                f"**Step {step['step_number']}: {step['agent']}**"
                                            )
                                            st.write(step["action"])
                                        with col2:
                                            if step.get("entropy_reduction"):
                                                st.metric(
                                                    "Reduction",
                                                    f"{step['entropy_reduction']:.1%}",
                                                )

                            break

                        elif result["status"] == "failed":
                            st.error(
                                f"❌ Query failed: {result.get('error', 'Unknown error')}"
                            )
                            break

                        else:
                            progress = min((attempt / max_attempts) * 100, 90)
                            progress_bar.progress(int(progress) / 100)
                            status_text.info(
                                f"⏳ Processing... (step {len(result.get('reasoning_steps', []))})"
                            )
                            time.sleep(1)
                    else:
                        st.warning("⚠️ Query timeout.")

                else:
                    st.error(f"API Error: {submit_response.status_code}")

            except Exception as e:
                st.error(f"Request failed: {str(e)}")

with browse_tab:
    st.subheader("📖 Browse Regulations")

    browse_section = st.selectbox(
        "Select Section",
        options=["Regulations", "Articles", "Chapters", "Definitions", "Recitals", "Annexes"],
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
            col1, col2 = st.columns(2)
            with col1:
                reg_select = st.selectbox("Select Regulation:", ["eu_ai_act", "dsa"])
            with col2:
                article_num = st.number_input("Article Number:", min_value=1, max_value=200)

            if st.button("📄 Load Article"):
                response = requests.get(
                    f"{API_BASE_URL}/article/{reg_select}/{int(article_num)}", timeout=10
                )
                if response.status_code == 200:
                    article = response.json()
                    st.write(f"**Title:** {article.get('title', 'N/A')}")
                    st.write(f"**Content:** {article.get('content', 'N/A')}")
                    if article.get("children"):
                        st.write("**Sub-sections:**")
                        for child in article["children"]:
                            st.write(f"  - {child.get('title', child.get('id'))}")
                else:
                    st.error(f"Article not found (Status: {response.status_code})")

        elif browse_section == "Chapters":
            col1, col2 = st.columns(2)
            with col1:
                reg_select = st.selectbox("Select Regulation:", ["eu_ai_act", "dsa"])
            with col2:
                chapter_num = st.text_input("Chapter Number:")

            if st.button("📚 Load Chapter") and chapter_num:
                response = requests.get(
                    f"{API_BASE_URL}/chapter/{reg_select}/{chapter_num}", timeout=10
                )
                if response.status_code == 200:
                    chapter = response.json()
                    st.write(f"**Title:** {chapter.get('title', 'N/A')}")
                    if chapter.get("children"):
                        st.write("**Articles in Chapter:**")
                        for child in chapter["children"]:
                            st.write(f"  - Article {child.get('number')}: {child.get('title', 'N/A')}")
                else:
                    st.error(f"Chapter not found (Status: {response.status_code})")

        elif browse_section == "Definitions":
            reg_select = st.selectbox("Select Regulation:", ["eu_ai_act", "dsa"])

            response = requests.get(f"{API_BASE_URL}/definitions/{reg_select}", timeout=10)
            if response.status_code == 200:
                definitions = response.json()
                for defn in definitions[:20]:
                    st.write(f"**{defn.get('term', 'Unknown')}:** {defn.get('definition', 'N/A')}")
            else:
                st.error("Failed to load definitions")

        elif browse_section == "Recitals":
            reg_select = st.selectbox("Select Regulation:", ["eu_ai_act", "dsa"])

            response = requests.get(f"{API_BASE_URL}/recitals/{reg_select}", timeout=10)
            if response.status_code == 200:
                recitals = response.json()
                for recital in recitals[:20]:
                    st.write(f"**Recital {recital.get('number')}:** {recital.get('text', 'N/A')[:150]}...")
            else:
                st.error("Failed to load recitals")

        elif browse_section == "Annexes":
            reg_select = st.selectbox("Select Regulation:", ["eu_ai_act", "dsa"])

            response = requests.get(f"{API_BASE_URL}/annexes/{reg_select}", timeout=10)
            if response.status_code == 200:
                annexes = response.json()
                for annex in annexes:
                    st.write(f"**Annex {annex.get('number')}:** {annex.get('title', 'N/A')}")
            else:
                st.error("Failed to load annexes")

    except Exception as e:
        st.error(f"Error: {str(e)}")

with history_tab:
    st.subheader("Query History")

    if not st.session_state.query_history:
        st.info("No queries yet. Submit a question in the Query tab!")
    else:
        for idx, query in enumerate(st.session_state.query_history):
            with st.expander(f"🕐 {query['question'][:70]}..."):
                st.write(f"**Status:** {query['status']}")
                st.write(f"**Time:** {query.get('timestamp', 'N/A')}")
                if query.get("final_answer"):
                    st.write(f"**Answer Preview:** {query['final_answer'][:150]}...")

with about_tab:
    st.markdown(
        """
        ## About CORTEX-RAG

        **CORTEX-RAG** is a High-Precision Regulatory Discovery System for navigating
        the **EU AI Act** and **Digital Services Act (DSA)**.

        ### Key Innovations

        💡 **GraphRAG Architecture** — Navigate legal dependencies through a Neo4j knowledge graph

        🔍 **Semantic Entropy Pruning** — Reduce context by ~50% without sacrificing accuracy

        🤖 **Agentic Self-Correction** — Verify and re-retrieve context in real-time

        ### Technical Stack

        - **LLM:** Llama 3.1 8B
        - **Graph DB:** Neo4j
        - **Frontend:** Streamlit
        - **API:** FastAPI
        - **Infrastructure:** Docker + Nuvelos

        ### Team

        - Data Architect: Knowledge Graph Engineering
        - LLM Engineer: Reasoning Engine & Entropy Pruning
        - Full-Stack Developer: Web UI & API Gateway
        - Systems Lead: Deployment & Benchmarking

        ---

        *Built with ⚡ for digital sustainability and regulatory compliance*
        """
    )
