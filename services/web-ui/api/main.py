"""FastAPI application - Web UI API Gateway."""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReasoningStep,
)

# In-memory query store (replace with Redis in production)
query_store: dict[str, QueryResponse] = {}

# Service URLs from environment
REASONING_ENGINE_URL = "http://127.0.0.1:8002"
# REASONING_ENGINE_URL = os.getenv("REASONING_ENGINE_URL", "http://localhost:8002")
KNOWLEDGE_GRAPH_URL = os.getenv("KNOWLEDGE_GRAPH_URL", "http://localhost:8001")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # Startup
    app.state.http_client = httpx.AsyncClient(timeout=None)
    yield
    # Shutdown
    await app.state.http_client.aclose()


app = FastAPI(
    title="CORTEX-RAG Web UI API",
    description="Gateway API for Regulatory Compliance Dashboard",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check health of this service and dependencies."""
    reasoning_available = False
    kg_available = False

    try:
        response = await app.state.http_client.get(f"{REASONING_ENGINE_URL}/health")
        reasoning_available = response.status_code == 200
    except Exception:
        pass

    try:
        response = await app.state.http_client.get(f"{KNOWLEDGE_GRAPH_URL}/health")
        kg_available = response.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if reasoning_available and kg_available else "degraded",
        reasoning_engine_available=reasoning_available,
        knowledge_graph_available=kg_available,
        timestamp=datetime.now(),
    )


@app.post("/api/query")
async def submit_query(request: QueryRequest) -> dict:
    """Submit a legal query for processing."""
    query_id = str(uuid4())

    # Initialize query in store
    query_store[query_id] = QueryResponse(
        query_id=query_id,
        status="processing",
        question=request.question,
        reasoning_steps=[],
        graph_data={"nodes": [], "edges": []},
        citations=[],
        metrics={},
    )

    # Forward to reasoning engine asynchronously
    task = asyncio.create_task(process_query(query_id, request))
    app.state.tasks = getattr(app.state, "tasks", {})
    app.state.tasks[query_id] = task

    return {"query_id": query_id, "status": "processing"}


async def process_query(query_id: str, request: QueryRequest) -> None:
    """Process query by forwarding to reasoning engine."""
    try:
        started = datetime.now()

        kg_search = await _fetch_kg_search(request)
        graph_data = await _build_graph_from_kg(kg_search, request)

        reasoning_response = await _try_reasoning_engine(request, kg_search)
        if reasoning_response:
            result = reasoning_response
            result.setdefault("graph_data", graph_data)
            result.setdefault("citations", _build_citations_from_kg(kg_search))
            result.setdefault(
                "metrics",
                {
                    "reasoning_steps": len(result.get("reasoning_steps", [])),
                    "nodes_retrieved": len(graph_data.get("nodes", [])),
                },
            )
        else:
            result = _build_local_fallback_result(request, kg_search, graph_data, started)

        query_store[query_id].status = "completed"
        query_store[query_id].final_answer = result.get("final_answer")
        query_store[query_id].reasoning_steps = [
            ReasoningStep(**step) for step in result.get("reasoning_steps", [])
        ]
        query_store[query_id].graph_data = result.get("graph_data", {})
        query_store[query_id].citations = result.get("citations", [])
        query_store[query_id].metrics = result.get("metrics", {})

    except Exception as e:
        query_store[query_id].status = "failed"
        query_store[query_id].error = str(e)


async def _fetch_kg_search(request: QueryRequest) -> list[dict[str, Any]]:
    """Retrieve relevant legal nodes from the knowledge-graph service."""
    params: dict[str, Any] = {"q": request.question, "limit": 12}
    if request.regulation != "both":
        params["regulation"] = request.regulation

    response = await app.state.http_client.get(f"{KNOWLEDGE_GRAPH_URL}/graph/search", params=params)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []


async def _build_graph_from_kg(
    kg_search: list[dict[str, Any]],
    request: QueryRequest,
) -> dict[str, list[dict[str, Any]]]:
    """Transform KG search + traversal responses into graph nodes/edges for UI."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str], dict[str, Any]] = {}

    top_hits = kg_search[: min(5, len(kg_search))]
    for hit in top_hits:
        node = hit.get("result", {})
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue

        label = _display_label(node_id=node_id, node=node)
        node_type = str(hit.get("label") or _infer_node_type(node_id))
        preview = str(node.get("summary") or node.get("full_text") or node.get("text") or "")
        nodes[node_id] = {
            "id": node_id,
            "label": label,
            "node_type": node_type,
            "text_preview": preview[:220] if preview else None,
            "entropy_score": float(hit.get("score", 0.0)),
            "pruned": False,
            "regulation": str(node.get("regulation") or request.regulation),
        }

        try:
            traversal = await app.state.http_client.get(
                f"{KNOWLEDGE_GRAPH_URL}/graph/traverse/{node_id}",
                params={"depth": request.max_hops, "direction": "both"},
            )
            traversal.raise_for_status()
            payload = traversal.json()
            source = payload.get("source", {})
            src_id = str(source.get("id", node_id))

            for neighbor in payload.get("neighbors", []):
                n_id = str(neighbor.get("id", "")).strip()
                if not n_id:
                    continue
                if n_id not in nodes:
                    n_type = str(neighbor.get("label") or _infer_node_type(n_id))
                    n_title = str(neighbor.get("title") or "")
                    nodes[n_id] = {
                        "id": n_id,
                        "label": n_title[:60] if n_title else n_id,
                        "node_type": n_type,
                        "text_preview": n_title[:220] if n_title else None,
                        "entropy_score": None,
                        "pruned": False,
                        "regulation": str(request.regulation),
                    }

                edge_key = tuple(sorted((src_id, n_id)))
                if edge_key not in edges:
                    edges[edge_key] = {
                        "source": src_id,
                        "target": n_id,
                        "relationship": ":CONNECTED",
                        "strength": 1.0,
                    }
        except Exception:
            # Keep primary search nodes even if traversal fails for some hits.
            continue

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


async def _try_reasoning_engine(
    request: QueryRequest,
    kg_search: list[dict[str, Any]],
) -> dict[str, Any] | None:
    payload = {
        "question": request.question,
        "regulation": request.regulation,
        "max_hops": request.max_hops,
        "enable_pruning": request.enable_pruning,
        "enable_self_correction": request.enable_self_correction,
        "pruning_threshold": request.pruning_threshold,
        "kg_hits": kg_search,
    }

    try:
        response = await app.state.http_client.post(
            "http://127.0.0.1:8002/api/reason",  # Use 127.0.0.1 directly
            json=payload,
            timeout=None,  # Give it enough room
        )
        if response.status_code >= 400:
            print(f"❌ Reasoning Engine returned error: {response.status_code}")
            return None

        return response.json()
    except Exception as e:
        print(f"❌ Connection to Reasoning Engine failed: {e!s}")
        return None


def _build_local_fallback_result(
    request: QueryRequest,
    kg_search: list[dict[str, Any]],
    graph_data: dict[str, list[dict[str, Any]]],
    started: datetime,
) -> dict[str, Any]:
    """Build a deterministic fallback when reasoning-engine is not yet integrated."""
    retrieved_ids = [str(item.get("result", {}).get("id", "")).strip() for item in kg_search]
    retrieved_ids = [node_id for node_id in retrieved_ids if node_id][:6]

    top_summaries: list[str] = []
    for item in kg_search[:3]:
        result = item.get("result", {})
        snippet = str(
            result.get("summary") or result.get("full_text") or result.get("text") or ""
        ).strip()
        if snippet:
            top_summaries.append(snippet[:220])

    summary_text = (
        " ".join(top_summaries)
        if top_summaries
        else "No relevant legal summary returned by KG search."
    )
    elapsed = (datetime.now() - started).total_seconds()

    return {
        "final_answer": (
            f"KG retrieved {len(retrieved_ids)} relevant nodes for your question. "
            f"Preliminary context: {summary_text}"
        ),
        "reasoning_steps": [
            {
                "step_number": 1,
                "agent": "Retriever",
                "action": "Queried knowledge-graph /graph/search and /graph/traverse",
                "retrieved_nodes": retrieved_ids,
                "entropy_reduction": 0.0,
                "timestamp": datetime.now().isoformat(),
            },
            {
                "step_number": 2,
                "agent": "Synthesizer",
                "action": "Built interim answer while reasoning-engine endpoint is unavailable",
                "retrieved_nodes": retrieved_ids[:3],
                "entropy_reduction": 0.0,
                "timestamp": datetime.now().isoformat(),
            },
        ],
        "graph_data": graph_data,
        "citations": _build_citations_from_kg(kg_search),
        "metrics": {
            "reasoning_steps": 2,
            "entropy_reduction_percent": 0,
            "tokens_saved": 0,
            "latency_seconds": round(elapsed, 3),
            "nodes_pruned": 0,
            "nodes_retrieved": len(graph_data.get("nodes", [])),
        },
    }


def _build_citations_from_kg(kg_search: list[dict[str, Any]]) -> list[str]:
    citations: list[str] = []
    for item in kg_search[:5]:
        node = item.get("result", {})
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        title = str(node.get("title") or node.get("term") or node_id)
        regulation = str(node.get("regulation") or "unknown")
        citations.append(f"{title} ({regulation}) - {node_id}")
    return citations


def _display_label(node_id: str, node: dict[str, Any]) -> str:
    if node.get("title"):
        return str(node["title"])[:60]
    if node.get("term"):
        return str(node["term"])[:60]
    if node.get("number") is not None:
        return f"{_infer_node_type(node_id)} {node.get('number')}"
    return node_id[:60]


def _infer_node_type(node_id: str) -> str:
    if "_art_" in node_id:
        return "Article"
    if "_rec_" in node_id:
        return "Recital"
    if "_def_" in node_id:
        return "Definition"
    if "_par_" in node_id:
        return "Paragraph"
    return "Node"


@app.get("/api/query/{query_id}", response_model=QueryResponse)
async def get_query_status(query_id: str) -> QueryResponse:
    """Get status and results of a query."""
    if query_id not in query_store:
        raise HTTPException(status_code=404, detail="Query not found")

    return query_store[query_id]


@app.get("/api/graph/{query_id}")
async def get_graph_visualization(query_id: str) -> dict:
    """Get graph data formatted for visualization."""
    if query_id not in query_store:
        raise HTTPException(status_code=404, detail="Query not found")

    query = query_store[query_id]

    # Transform to Cytoscape format
    cytoscape_elements = []

    # Add nodes
    for node in query.graph_data.get("nodes", []):
        cytoscape_elements.append(
            {
                "data": {
                    "id": node["id"],
                    "label": node["label"],
                    "type": node["node_type"],
                    "pruned": node.get("pruned", False),
                    "entropy": node.get("entropy_score"),
                },
                "classes": "pruned" if node.get("pruned") else "",
            }
        )

    # Add edges
    for edge in query.graph_data.get("edges", []):
        cytoscape_elements.append(
            {
                "data": {
                    "source": edge["source"],
                    "target": edge["target"],
                    "relationship": edge["relationship"],
                    "strength": edge.get("strength", 1.0),
                }
            }
        )

    return {"elements": cytoscape_elements}


@app.delete("/api/query/{query_id}")
async def delete_query(query_id: str) -> dict:
    """Delete a query from the store."""
    if query_id in query_store:
        del query_store[query_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Query not found")


# ── Knowledge Graph Browse Endpoints ────────────────────────────────────────


@app.get("/api/regulations")
async def list_regulations() -> list[dict]:
    """List all available regulations from the knowledge graph."""
    try:
        response = await app.state.http_client.get(f"{KNOWLEDGE_GRAPH_URL}/graph/regulations")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch regulations: {e!s}") from e


@app.get("/api/article/{regulation}/{number}")
async def get_article(regulation: str, number: int) -> dict:
    """Get a specific article by regulation and number."""
    try:
        response = await app.state.http_client.get(
            f"{KNOWLEDGE_GRAPH_URL}/graph/article/{regulation}/{number}"
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=404, detail=f"Article {number} not found in {regulation}"
            ) from e
        raise HTTPException(status_code=503, detail="Knowledge graph service error") from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"{e!s}") from e


@app.get("/api/chapter/{regulation}/{number}")
async def get_chapter(regulation: str, number: str) -> dict:
    """Get a chapter with its articles."""
    try:
        response = await app.state.http_client.get(
            f"{KNOWLEDGE_GRAPH_URL}/graph/chapter/{regulation}/{number}"
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=404, detail=f"Chapter {number} not found in {regulation}"
            ) from e
        raise HTTPException(status_code=503, detail="Knowledge graph service error") from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"{e!s}") from e


@app.get("/api/recitals/{regulation}")
async def list_recitals(regulation: str, skip: int = 0, limit: int = 20) -> list[dict]:
    """List recitals for a regulation."""
    try:
        response = await app.state.http_client.get(
            f"{KNOWLEDGE_GRAPH_URL}/graph/recitals/{regulation}",
            params={"skip": skip, "limit": limit},
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch recitals: {e!s}") from e


@app.get("/api/annexes/{regulation}")
async def list_annexes(regulation: str) -> list[dict]:
    """List annexes for a regulation."""
    try:
        response = await app.state.http_client.get(
            f"{KNOWLEDGE_GRAPH_URL}/graph/annexes/{regulation}"
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch annexes: {e!s}") from e


@app.get("/api/definitions/{regulation}")
async def list_definitions(regulation: str) -> list[dict]:
    """List definitions for a regulation."""
    try:
        response = await app.state.http_client.get(
            f"{KNOWLEDGE_GRAPH_URL}/graph/definitions/{regulation}"
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch definitions: {e!s}") from e


@app.get("/api/stats")
async def get_graph_stats() -> dict:
    """Get graph statistics."""
    try:
        response = await app.state.http_client.get(f"{KNOWLEDGE_GRAPH_URL}/graph/stats/simple")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch stats: {e!s}") from e


@app.post("/api/cypher")
async def execute_cypher(query_payload: dict[str, Any]) -> dict:
    """Execute a read-only Cypher query on the knowledge graph."""
    try:
        payload = {
            "query": query_payload.get("query", ""),
            "parameters": query_payload.get("parameters", {}),
        }
        response = await app.state.http_client.post(
            f"{KNOWLEDGE_GRAPH_URL}/graph/cypher", json=payload
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cypher execution failed: {e!s}") from e


@app.post("/api/ingest")
async def ingest_data(ingest_payload: dict[str, Any]) -> dict:
    """Ingest enriched JSON data into the knowledge graph (admin endpoint)."""
    try:
        payload = {
            "file": ingest_payload.get("file"),
            "clear": ingest_payload.get("clear", False),
        }
        response = await app.state.http_client.post(
            f"{KNOWLEDGE_GRAPH_URL}/graph/ingest", json=payload
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ingest failed: {e!s}") from e


@app.websocket("/ws/{query_id}")
async def websocket_endpoint(websocket: WebSocket, query_id: str) -> None:
    """WebSocket for real-time updates on query processing."""
    await websocket.accept()

    try:
        while True:
            if query_id in query_store:
                await websocket.send_json(query_store[query_id].dict())

                if query_store[query_id].status in ["completed", "failed"]:
                    break

            await asyncio.sleep(0.5)  # Poll every 500ms

    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
