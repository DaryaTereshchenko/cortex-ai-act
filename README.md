# CORTEX-AI-ACT

> **EU AI Act Compliance Assistant** — Knowledge-graph-augmented reasoning with entropy pruning and self-correction.

---

## Architecture

| Service | Port | Description |
|---------|------|-------------|
| **Neo4j** | 7474 / 7687 | Knowledge Graph database (EU AI Act articles as nodes/edges) |
| **Knowledge-Graph** | 8001 | PDF→Markdown parsing, Neo4j schema management, Cypher query API |
| **Reasoning-Engine** | 8002 | Llama 3.1 8B + LangGraph — Entropy Pruning & Self-Correction Loop |
| **Web-UI** | 8501 | Streamlit dashboard with graph visualization |
| **Benchmarking** | 8003 | Token-reduction, latency, sustainability metrics |
| **NGINX** | 80 | Reverse proxy (production profile only) |

