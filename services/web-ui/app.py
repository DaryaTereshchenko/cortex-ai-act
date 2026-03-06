"""Web-UI — Streamlit entry point for the CORTEX full-stack application."""

from __future__ import annotations

import os

import httpx
import streamlit as st

st.set_page_config(page_title="CORTEX AI-Act", page_icon="🧠", layout="wide")

KG_URL = os.getenv("KG_SERVICE_URL", "http://localhost:8001")
RE_URL = os.getenv("RE_SERVICE_URL", "http://localhost:8002")

# ── Page layout ──────────────────────────────────────────────────────────────

st.title("🧠 CORTEX — AI Act Compliance Platform")

st.markdown(
    """
    Welcome to **CORTEX**, the AI Act compliance analysis platform.

    Use the services below to explore regulation data and run compliance queries.
    """
)

# ── Service health checks ────────────────────────────────────────────────────

st.subheader("Service Status")

col_kg, col_re = st.columns(2)

with col_kg:
    try:
        resp = httpx.get(f"{KG_URL}/health", timeout=5.0)
        data = resp.json()
        status = data.get("status", "unknown")
        if status == "ok":
            st.success("Knowledge Graph: connected")
        else:
            st.warning(f"Knowledge Graph: {status}")
    except Exception:
        st.error("Knowledge Graph: unreachable")

with col_re:
    try:
        resp = httpx.get(f"{RE_URL}/health", timeout=5.0)
        data = resp.json()
        status = data.get("status", "unknown")
        if status == "ok":
            st.success("Reasoning Engine: connected")
        else:
            st.warning(f"Reasoning Engine: {status}")
    except Exception:
        st.error("Reasoning Engine: unreachable")
