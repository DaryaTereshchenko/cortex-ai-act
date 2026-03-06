"""Knowledge-Graph UI — Streamlit explorer for the regulation knowledge graph."""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

st.set_page_config(page_title="Knowledge Graph Explorer", page_icon="🧠", layout="wide")

KG_URL = os.getenv("KG_SERVICE_URL", "http://localhost:8001")

# ── Helpers ──────────────────────────────────────────────────────────────────


def run_cypher(query: str) -> list[dict]:
    """Send a Cypher query to the knowledge-graph service."""
    try:
        resp = httpx.post(
            f"{KG_URL}/graph/cypher",
            json={"query": query},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json().get("rows", [])
    except Exception as exc:
        st.error(f"Knowledge-graph query failed: {exc}")
        return []


# ── Colour / size palette per node label ─────────────────────────────────────

_PALETTE: dict[str, dict[str, Any]] = {
    "Regulation": {"color": "#FF6B6B", "size": 50, "shape": "diamond"},
    "Chapter": {"color": "#FFA96B", "size": 35, "shape": "dot"},
    "Section": {"color": "#FFD93D", "size": 28, "shape": "dot"},
    "Article": {"color": "#6BCB77", "size": 22, "shape": "dot"},
    "Paragraph": {"color": "#4D96FF", "size": 14, "shape": "dot"},
    "SubParagraph": {"color": "#9B72FF", "size": 10, "shape": "dot"},
    "Definition": {"color": "#FF6BCC", "size": 18, "shape": "triangle"},
    "Recital": {"color": "#A8DADC", "size": 12, "shape": "dot"},
    "Annex": {"color": "#E9C46A", "size": 20, "shape": "square"},
    "AnnexSection": {"color": "#E9C46A", "size": 14, "shape": "square"},
}

_EDGE_COLORS: dict[str, str] = {
    "CONTAINS": "#555555",
    "PART_OF": "#555555",
    "NEXT": "#888888",
    "PREVIOUS": "#888888",
    "REFERENCES": "#2196F3",
    "OVERLAPS": "#FF5722",
    "SIMILAR_TERM": "#FF9800",
    "RELATED_REGULATION": "#F44336",
}

# ── Prebuilt view queries ────────────────────────────────────────────────────

_VIEWS: dict[str, dict[str, str]] = {
    "Full hierarchy - EU AI Act": {
        "description": "Regulation → Chapters → Sections/Articles (top two levels)",
        "query": """
            MATCH (r:Regulation {id: "eu_ai_act"})-[:CONTAINS]->(ch:Chapter)
            OPTIONAL MATCH (ch)-[:CONTAINS]->(child)
            RETURN r, ch, child
        """,
    },
    "Full hierarchy - DSA": {
        "description": "Regulation → Chapters → Sections/Articles",
        "query": """
            MATCH (r:Regulation {id: "dsa"})-[:CONTAINS]->(ch:Chapter)
            OPTIONAL MATCH (ch)-[:CONTAINS]->(child)
            RETURN r, ch, child
        """,
    },
    "Cross-regulation overlaps": {
        "description": "Articles linked by OVERLAPS between EU AI Act and DSA",
        "query": """
            MATCH (a:Article)-[o:OVERLAPS]->(b:Article)
            WHERE a.regulation = "eu_ai_act" AND b.regulation = "dsa"
            RETURN a, b
        """,
    },
    "Definitions - EU AI Act": {
        "description": "Article 3 → all 68 definitions",
        "query": """
            MATCH (a:Article {id: "eu_ai_act_art_3"})-[:CONTAINS]->(d:Definition)
            RETURN a, d
        """,
    },
    "Definitions - DSA": {
        "description": "Article 3 → all definitions (with sub-items)",
        "query": """
            MATCH (a:Article {id: "dsa_art_3"})-[:CONTAINS]->(d:Definition)
            OPTIONAL MATCH (d)-[:CONTAINS]->(sp:SubParagraph)
            RETURN a, d, sp
        """,
    },
    "Deep dive - one chapter": {
        "description": "EU AI Act Chapter III fully expanded (sections → articles → paragraphs)",
        "query": """
            MATCH path = (ch:Chapter {id: "eu_ai_act_ch_III"})-[:CONTAINS*1..3]->(desc)
            RETURN ch, desc
        """,
    },
    "Similar definitions (cross-reg)": {
        "description": "Definition pairs linked by SIMILAR_TERM",
        "query": """
            MATCH (a:Definition)-[s:SIMILAR_TERM]->(b:Definition)
            WHERE a.regulation = "eu_ai_act"
            RETURN a, b
        """,
    },
    "Both regulations - top level": {
        "description": "Both regulations with all their chapters side by side",
        "query": """
            MATCH (r:Regulation)-[:CONTAINS]->(ch:Chapter)
            RETURN r, ch
        """,
    },
}

# ── Graph-building logic ─────────────────────────────────────────────────────


def build_graph_from_view(view_key: str) -> tuple[list[Node], list[Edge]]:
    """Run the selected view and return agraph Nodes + Edges."""
    view = _VIEWS[view_key]

    # Use a path-returning query to get both nodes and relationships
    node_query = f"""
        CALL {{
            {view["query"].strip().rstrip(";")}
        }}
        WITH *
        UNWIND [x IN [
            {_collect_vars(view_key)}
        ] | x] AS node
        WITH node WHERE node IS NOT NULL
        RETURN DISTINCT
            elementId(node) AS eid,
            node.id AS nid,
            labels(node)[0] AS label,
            CASE labels(node)[0]
                WHEN 'Regulation' THEN node.short_name
                WHEN 'Chapter'    THEN 'Ch ' + node.number + ': ' + substring(node.title, 0, 30)
                WHEN 'Section'    THEN 'Sec ' + node.number
                WHEN 'Article'    THEN 'Art ' + toString(node.number)
                WHEN 'Paragraph'  THEN '¶' + node.number
                WHEN 'SubParagraph' THEN '¶' + node.number
                WHEN 'Definition' THEN node.term
                WHEN 'Recital'    THEN 'Rec ' + toString(node.number)
                WHEN 'Annex'      THEN 'Annex ' + node.number
                WHEN 'AnnexSection' THEN 'AnxSec ' + node.number
                ELSE coalesce(node.id, toString(elementId(node)))
            END AS display,
            node.regulation AS regulation
    """

    edge_query = f"""
        CALL {{
            {view["query"].strip().rstrip(";")}
        }}
        WITH *
        UNWIND [x IN [
            {_collect_vars(view_key)}
        ] | x] AS node
        WITH collect(DISTINCT node.id) AS node_ids
        MATCH (a)-[r]->(b)
        WHERE a.id IN node_ids AND b.id IN node_ids
           AND type(r) IN ['CONTAINS', 'OVERLAPS', 'SIMILAR_TERM', 'RELATED_REGULATION', 'REFERENCES']
        RETURN DISTINCT a.id AS src, b.id AS tgt, type(r) AS rel
    """

    nodes_data = run_cypher(node_query)
    edges_data = run_cypher(edge_query)

    if not nodes_data:
        return [], []

    nodes = []
    seen_ids: set[str] = set()
    for row in nodes_data:
        nid = row.get("nid", "")
        if not nid or nid in seen_ids:
            continue
        seen_ids.add(nid)
        label = row.get("label", "")
        style = _PALETTE.get(label, {"color": "#999", "size": 15, "shape": "dot"})
        reg = row.get("regulation", "")
        # Slightly different shade for DSA nodes
        color = style["color"]
        if reg == "dsa" and label not in ("Regulation",):
            color = _adjust_hue(color)

        nodes.append(
            Node(
                id=nid,
                label=row.get("display", nid)[:40],
                size=style["size"],
                color=color,
                shape=style["shape"],
                title=f"{label}: {row.get('display', nid)}\nRegulation: {reg}\nID: {nid}",
            )
        )

    edges = []
    seen_edges: set[tuple] = set()
    for row in edges_data:
        src, tgt, rel = row.get("src"), row.get("tgt"), row.get("rel", "")
        if not src or not tgt:
            continue
        if src not in seen_ids or tgt not in seen_ids:
            continue
        key = (src, tgt, rel)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append(
            Edge(
                source=src,
                target=tgt,
                label=rel
                if rel in ("OVERLAPS", "SIMILAR_TERM", "REFERENCES", "RELATED_REGULATION")
                else "",
                color=_EDGE_COLORS.get(rel, "#666"),
                width=3 if rel in ("OVERLAPS", "SIMILAR_TERM") else 1,
            )
        )

    return nodes, edges


def _collect_vars(view_key: str) -> str:
    """Return the Cypher list of variable names used in each view query."""
    mapping = {
        "Full hierarchy - EU AI Act": "r, ch, child",
        "Full hierarchy - DSA": "r, ch, child",
        "Cross-regulation overlaps": "a, b",
        "Definitions - EU AI Act": "a, d",
        "Definitions - DSA": "a, d, sp",
        "Deep dive - one chapter": "ch, desc",
        "Similar definitions (cross-reg)": "a, b",
        "Both regulations - top level": "r, ch",
    }
    return mapping.get(view_key, "r, ch")


def _adjust_hue(hex_color: str) -> str:
    """Lighten a hex color slightly for DSA regulation distinction."""
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        r = min(255, r + 40)
        g = min(255, g + 40)
        b = min(255, b + 40)
        return f"#{r:02x}{g:02x}{b:02x}"
    except (ValueError, IndexError):
        return hex_color


# ── Stats dashboard ──────────────────────────────────────────────────────────


def show_stats():
    """Display key graph statistics."""
    counts = run_cypher("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC")
    rels = run_cypher("MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC")

    if counts:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Node counts")
            for row in counts:
                label = row["label"]
                cnt = row["cnt"]
                style = _PALETTE.get(label, {})
                color = style.get("color", "#999")
                st.markdown(
                    f"<span style='color:{color}; font-weight:bold'>●</span> **{label}**: {cnt:,}",
                    unsafe_allow_html=True,
                )
        with col2:
            st.subheader("Relationship counts")
            for row in rels:
                rel = row["rel"]
                cnt = row["cnt"]
                color = _EDGE_COLORS.get(rel, "#666")
                st.markdown(
                    f"<span style='color:{color}; font-weight:bold'>—</span> **{rel}**: {cnt:,}",
                    unsafe_allow_html=True,
                )


# ── Page layout ──────────────────────────────────────────────────────────────

st.title("🧠 Knowledge Graph Explorer")

tab_explore, tab_stats, tab_cypher = st.tabs(
    ["📊 Graph Explorer", "📈 Statistics", "🔍 Cypher Query"]
)

# ── Tab 1: Graph Explorer ────────────────────────────────────────────────────
with tab_explore:
    st.markdown("Select a **preset view** to visualize the knowledge graph structure.")

    view_key = st.selectbox(
        "Select view",
        options=list(_VIEWS.keys()),
        format_func=lambda k: f"{k} — {_VIEWS[k]['description']}",
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        physics_enabled = st.checkbox("Physics simulation", value=True)
    with col_cfg2:
        hierarchical = st.checkbox("Hierarchical layout", value=False)
    with col_cfg3:
        height = st.slider("Graph height (px)", 400, 1000, 700, step=50)

    if st.button("🔄 Render graph", type="primary", use_container_width=True):
        with st.spinner("Querying knowledge graph…"):
            nodes, edges = build_graph_from_view(view_key)

        if nodes:
            st.success(f"Showing **{len(nodes)} nodes** and **{len(edges)} edges**")

            config = Config(
                width="100%",
                height=height,
                directed=True,
                physics=physics_enabled,
                hierarchical=hierarchical,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=True,
                node={"labelProperty": "label"},
                link={"labelProperty": "label", "renderLabel": True},
            )

            agraph(nodes=nodes, edges=edges, config=config)

            # Legend
            st.markdown("---")
            st.markdown("**Legend:**")
            legend_cols = st.columns(5)
            for i, (label, style) in enumerate(_PALETTE.items()):
                with legend_cols[i % 5]:
                    st.markdown(
                        f"<span style='color:{style['color']}; font-size:20px'>●</span> {label}",
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("No nodes returned. The knowledge graph may not be ingested yet.")

# ── Tab 2: Statistics ────────────────────────────────────────────────────────
with tab_stats:
    st.markdown("Overview of the knowledge graph contents.")
    show_stats()

    st.markdown("---")
    st.subheader("Hierarchy summary")
    hierarchy = run_cypher("""
        MATCH (r:Regulation)
        OPTIONAL MATCH (r)-[:CONTAINS]->(ch:Chapter)
        WITH r, count(ch) AS chapters
        OPTIONAL MATCH (a:Article) WHERE a.regulation = r.id
        WITH r, chapters, count(a) AS articles
        OPTIONAL MATCH (d:Definition) WHERE d.regulation = r.id
        WITH r, chapters, articles, count(d) AS definitions
        OPTIONAL MATCH (p:Paragraph) WHERE p.regulation = r.id
        WITH r, chapters, articles, definitions, count(p) AS paragraphs
        OPTIONAL MATCH (sp:SubParagraph) WHERE sp.regulation = r.id
        RETURN r.short_name AS regulation, chapters, articles,
               definitions, paragraphs, count(sp) AS subparagraphs
    """)
    if hierarchy:
        import pandas as pd

        st.dataframe(pd.DataFrame(hierarchy), use_container_width=True, hide_index=True)

# ── Tab 3: Custom Cypher ─────────────────────────────────────────────────────
with tab_cypher:
    st.markdown("Run any **read-only** Cypher query against the knowledge graph.")
    custom_query = st.text_area(
        "Cypher query",
        value="MATCH (r:Regulation)-[:CONTAINS]->(ch:Chapter) RETURN r.short_name AS regulation, ch.number AS chapter, ch.title AS title ORDER BY r.short_name, ch.number",
        height=120,
    )

    if st.button("▶ Run query", use_container_width=True) and custom_query.strip():
        # Basic safety: block write operations
        upper = custom_query.upper().strip()
        if any(
            kw in upper for kw in ["DELETE", "DETACH", "CREATE", "SET ", "REMOVE", "MERGE", "DROP"]
        ):
            st.error("Only read queries are allowed in this interface.")
        else:
            with st.spinner("Running query…"):
                results = run_cypher(custom_query)
            if results:
                import pandas as pd

                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                st.caption(f"{len(results)} rows returned")
            else:
                st.info("Query returned no results.")
