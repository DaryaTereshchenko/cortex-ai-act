"""Web-UI — Streamlit entry point."""

from __future__ import annotations

import os

import streamlit as st

st.set_page_config(page_title="CORTEX AI-Act", page_icon="🧠", layout="wide")

st.title("CORTEX — EU AI Act Compliance Assistant")
st.markdown(
    "Ask a question about the **EU AI Act** and watch the knowledge-graph reasoning in real time."
)

KG_URL = os.getenv("KG_SERVICE_URL", "http://localhost:8001")
RE_URL = os.getenv("RE_SERVICE_URL", "http://localhost:8002")

query = st.text_input("Enter your compliance question:")

if query:
    st.info("Reasoning engine integration coming soon…")
