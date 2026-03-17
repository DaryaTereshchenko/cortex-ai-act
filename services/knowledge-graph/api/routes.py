"""
API routes for the Knowledge-Graph service.

Provides endpoints for:
  * Ingesting enriched JSON data into Neo4j
  * Querying the graph (articles, chapters, recitals, search, traversal)
  * Running read-only Cypher queries for agent integration
  * Graph statistics
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from graph.builder import GraphBuilder
from graph.connection import Neo4jConnection

log = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["knowledge-graph"])

# ── Shared connection (initialised in main.py lifespan) ─────────────────────
_conn: Neo4jConnection | None = None


def init_connection(conn: Neo4jConnection) -> None:
    """Called by the app lifespan to inject the shared connection."""
    global _conn
    _conn = conn


def _get_conn() -> Neo4jConnection:
    if _conn is None:
        raise HTTPException(503, "Neo4j connection not initialised")
    return _conn


# ── Request / response models ───────────────────────────────────────────────


class IngestRequest(BaseModel):
    file: str | None = Field(
        None,
        description="Specific enriched JSON filename (e.g. 'ai_act_extracted_enriched.json'). "
        "Omit to ingest all enriched files.",
    )
    clear: bool = Field(False, description="Clear the entire graph before ingestion")


class CypherRequest(BaseModel):
    query: str = Field(..., description="Cypher query (read-only)")
    parameters: dict[str, Any] = Field(default_factory=dict)


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.post("/ingest", summary="Ingest enriched JSON into Neo4j")
async def ingest(req: IngestRequest) -> dict:
    conn = _get_conn()
    builder = GraphBuilder(conn)

    if req.clear:
        builder.clear_graph()

    data_dir = Path(__file__).resolve().parent.parent / "data"

    if req.file:
        path = data_dir / req.file
        if not path.exists():
            raise HTTPException(404, f"File not found: {req.file}")
        stats = builder.ingest_file(path)
        return {"status": "ok", "file": req.file, "stats": stats}

    stats = builder.ingest_all(data_dir)
    return {"status": "ok", "files": list(stats.keys()), "stats": stats}


@router.get("/stats", summary="Graph statistics")
async def graph_stats() -> dict:
    conn = _get_conn()
    # Node counts by label
    labels_result = conn.execute_read("""
        CALL db.labels() YIELD label
        CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) AS cnt', {})
        YIELD value
        RETURN label, value.cnt AS count
        """)
    node_counts = {r["label"]: r["count"] for r in labels_result}

    # Relationship counts
    rel_result = conn.execute_read("""
        CALL db.relationshipTypes() YIELD relationshipType AS type
        CALL apoc.cypher.run(
            'MATCH ()-[r:`' + type + '`]->() RETURN count(r) AS cnt', {}
        ) YIELD value
        RETURN type, value.cnt AS count
        """)
    rel_counts = {r["type"]: r["count"] for r in rel_result}

    return {"nodes": node_counts, "relationships": rel_counts}


@router.get("/stats/simple", summary="Simple graph statistics (no APOC)")
async def graph_stats_simple() -> dict:
    conn = _get_conn()
    result = conn.execute_read("""
        MATCH (n)
        WITH labels(n) AS lbls, count(n) AS cnt
        UNWIND lbls AS label
        RETURN label, sum(cnt) AS count
        """)
    node_counts = {r["label"]: r["count"] for r in result}

    rel_result = conn.execute_read("""
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(r) AS count
        """)
    rel_counts = {r["type"]: r["count"] for r in rel_result}

    return {"nodes": node_counts, "relationships": rel_counts}


@router.get("/regulations", summary="List all regulations")
async def list_regulations() -> list[dict]:
    conn = _get_conn()
    return conn.execute_read("""
        MATCH (r:Regulation)
        RETURN r {.*} AS regulation
        ORDER BY r.id
        """)


@router.get(
    "/article/{regulation}/{number}",
    summary="Get article by regulation and number",
)
async def get_article(regulation: str, number: int) -> dict:
    conn = _get_conn()
    result = conn.execute_read(
        """
        MATCH (a:Article {regulation: $reg, number: $num})
        OPTIONAL MATCH (a)-[:PART_OF]->(parent)
        OPTIONAL MATCH (a)-[:CONTAINS]->(child)
        RETURN a {.*} AS article,
               labels(parent)[0] AS parent_label,
               parent.id AS parent_id,
               collect(DISTINCT {id: child.id, label: labels(child)[0], number: child.number}) AS children
        """,
        {"reg": regulation, "num": number},
    )
    if not result:
        raise HTTPException(404, f"Article {number} not found in {regulation}")
    return result[0]


@router.get(
    "/chapter/{regulation}/{number}",
    summary="Get chapter with its articles",
)
async def get_chapter(regulation: str, number: str) -> dict:
    conn = _get_conn()
    result = conn.execute_read(
        """
        MATCH (c:Chapter {regulation: $reg, number: $num})
        OPTIONAL MATCH (c)-[:CONTAINS]->(child)
        RETURN c {.*} AS chapter,
               collect(DISTINCT {id: child.id, label: labels(child)[0],
                      title: child.title, number: child.number}) AS children
        """,
        {"reg": regulation, "num": number},
    )
    if not result:
        raise HTTPException(404, f"Chapter {number} not found in {regulation}")
    return result[0]


@router.get(
    "/recitals/{regulation}",
    summary="List recitals for a regulation",
)
async def list_recitals(
    regulation: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
) -> list[dict]:
    conn = _get_conn()
    return conn.execute_read(
        """
        MATCH (rec:Recital {regulation: $reg})
        RETURN rec {.*} AS recital
        ORDER BY rec.number
        SKIP $skip LIMIT $limit
        """,
        {"reg": regulation, "skip": skip, "limit": limit},
    )


@router.get(
    "/annexes/{regulation}",
    summary="List annexes for a regulation",
)
async def list_annexes(regulation: str) -> list[dict]:
    conn = _get_conn()
    return conn.execute_read(
        """
        MATCH (a:Annex {regulation: $reg})
        OPTIONAL MATCH (a)-[:CONTAINS]->(sec:AnnexSection)
        RETURN a {.*} AS annex,
               collect(DISTINCT {id: sec.id, number: sec.number, title: sec.title}) AS sections
        ORDER BY a.number
        """,
        {"reg": regulation},
    )


@router.get(
    "/definitions/{regulation}",
    summary="List definitions for a regulation",
)
async def list_definitions(regulation: str) -> list[dict]:
    conn = _get_conn()
    return conn.execute_read(
        """
        MATCH (d:Definition {regulation: $reg})
        RETURN d {.*} AS definition
        ORDER BY d.term
        """,
        {"reg": regulation},
    )


@router.get("/search", summary="Full-text search across articles and paragraphs")
async def search(
    q: str = Query(..., min_length=2, description="Search query"),
    regulation: str | None = Query(None, description="Filter by regulation ID"),
    limit: int = Query(10, ge=1, le=50),
) -> list[dict]:
    conn = _get_conn()
    # Use full-text index for articles
    params: dict[str, Any] = {"query": q, "limit": limit}
    reg_filter = ""
    if regulation:
        reg_filter = "AND node.regulation = $reg"
        params["reg"] = regulation

    results = conn.execute_read(
        f"""
        CALL db.index.fulltext.queryNodes('article_fulltext', $query)
        YIELD node, score
        WHERE score > 0.5 {reg_filter}
        RETURN node {{.*}} AS result, labels(node)[0] AS label, score
        ORDER BY score DESC
        LIMIT $limit
        """,
        params,
    )

    # Also search paragraphs
    par_results = conn.execute_read(
        f"""
        CALL db.index.fulltext.queryNodes('paragraph_fulltext', $query)
        YIELD node, score
        WHERE score > 0.5 {reg_filter}
        RETURN node {{.*}} AS result, labels(node)[0] AS label, score
        ORDER BY score DESC
        LIMIT $limit
        """,
        params,
    )

    combined = results + par_results
    combined.sort(key=lambda x: x.get("score", 0), reverse=True)
    return combined[:limit]


@router.get(
    "/traverse/{node_id}",
    summary="Traverse from a node — returns neighbors",
)
async def traverse(
    node_id: str,
    direction: str = Query("both", pattern="^(in|out|both)$"),
    depth: int = Query(1, ge=1, le=3),
) -> dict:
    conn = _get_conn()
    if direction == "out":
        pattern = f"(n {{id: $id}})-[r*1..{depth}]->(m)"
    elif direction == "in":
        pattern = f"(m)-[r*1..{depth}]->(n {{id: $id}})"
    else:
        pattern = f"(n {{id: $id}})-[r*1..{depth}]-(m)"

    result = conn.execute_read(
        f"""
        MATCH {pattern}
        WITH n, m, r
        RETURN n {{.*}} AS source,
               collect(DISTINCT {{
                   id: m.id,
                   label: labels(m)[0],
                   title: m.title,
                   number: m.number
               }}) AS neighbors
        """,
        {"id": node_id},
    )
    if not result:
        raise HTTPException(404, f"Node {node_id} not found")
    return result[0]


_WRITE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s+\{)\b",
    re.IGNORECASE,
)


@router.post("/cypher", summary="Execute a read-only Cypher query")
async def run_cypher(req: CypherRequest) -> dict:
    """Execute a read-only Cypher query for agent integration."""
    if _WRITE_PATTERN.search(req.query):
        raise HTTPException(400, "Only read-only queries are allowed")
    conn = _get_conn()
    rows = conn.execute_read(req.query, req.parameters)
    return {"columns": list(rows[0].keys()) if rows else [], "rows": rows}
