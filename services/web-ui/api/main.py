"""FastAPI application - Web UI API Gateway."""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    GraphEdge,
    GraphNode,
    HealthResponse,
    LegalNode,
    QueryRequest,
    QueryResponse,
    ReasoningStep,
)

# In-memory query store (replace with Redis in production)
query_store: Dict[str, QueryResponse] = {}

# Service URLs from environment
REASONING_ENGINE_URL = os.getenv("REASONING_ENGINE_URL", "http://reasoning-engine:8001")
KNOWLEDGE_GRAPH_URL = os.getenv("KNOWLEDGE_GRAPH_URL", "http://knowledge-graph:8002")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # Startup
    app.state.http_client = httpx.AsyncClient(timeout=30.0)
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
    asyncio.create_task(process_query(query_id, request))

    return {"query_id": query_id, "status": "processing"}


async def process_query(query_id: str, request: QueryRequest) -> None:
    """Process query by forwarding to reasoning engine."""
    try:
        # Simulate processing time
        await asyncio.sleep(2)

        # Mock legal nodes from knowledge-graph (matches KG service schema)
        mock_legal_nodes = [
            {
                "id": "ai_act_art_6",
                "node_type": "Article",
                "content": "Article 6 - Classification rules for high-risk AI systems. Providers shall draw up documentation before placing high-risk AI systems on the market...",
                "regulation": "eu_ai_act",
                "metadata": {"chapter": "III", "section": "1", "url": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj#art6"},
                "entropy_score": 0.92,
            },
            {
                "id": "ai_act_rec_47",
                "node_type": "Recital",
                "content": "Recital 47 - High-risk AI systems should be designed in a way that ensures human oversight and control...",
                "regulation": "eu_ai_act",
                "metadata": {"interprets": "ai_act_art_6", "url": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj#rec47"},
                "entropy_score": 0.85,
            },
            {
                "id": "ai_act_def_3",
                "node_type": "Definition",
                "content": "AI system: a machine-based system designed to operate with some degree of autonomy...",
                "regulation": "eu_ai_act",
                "metadata": {"source": "Article 3"},
                "entropy_score": 0.78,
            },
            {
                "id": "dsa_art_24",
                "node_type": "Article",
                "content": "Article 24 - Systemic risks shall refer to risks that may have systemic implications...",
                "regulation": "dsa",
                "metadata": {"chapter": "IV", "section": "2"},
                "entropy_score": 0.65,
            },
        ]

        # Simulate reasoning engine output (matches reasoning-engine schema)
        result = {
            "final_answer": (
                "Based on Article 6 of the EU AI Act and Article 24 of the DSA, "
                "high-risk AI systems require classification documentation and systemic risk assessment. "
                "The system must maintain human oversight per Recital 47."
            ),
            "reasoning_steps": [
                {
                    "step_number": 1,
                    "agent": "Retriever",
                    "action": "Searched knowledge graph for 'high-risk AI classification'",
                    "retrieved_nodes": ["ai_act_art_6", "ai_act_rec_47", "ai_act_def_3"],
                    "entropy_reduction": 0.18,
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "step_number": 2,
                    "agent": "Critic",
                    "action": "Validated hierarchy - checked for interpretive gaps",
                    "retrieved_nodes": ["ai_act_rec_47"],
                    "entropy_reduction": 0.08,
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "step_number": 3,
                    "agent": "Synthesizer",
                    "action": "Generated final answer with citations",
                    "retrieved_nodes": ["ai_act_art_6", "ai_act_rec_47", "dsa_art_24"],
                    "entropy_reduction": 0.12,
                    "timestamp": datetime.now().isoformat(),
                },
            ],
            "graph_data": {
                "nodes": [
                    {
                        "id": "ai_act_art_6",
                        "label": "Article 6",
                        "node_type": "Article",
                        "text_preview": "Classification rules for high-risk AI systems. Providers shall draw up documentation...",
                        "entropy_score": 0.92,
                        "pruned": False,
                        "regulation": "eu_ai_act",
                    },
                    {
                        "id": "ai_act_rec_47",
                        "label": "Recital 47",
                        "node_type": "Recital",
                        "text_preview": "High-risk AI systems should be designed in a way that ensures human oversight...",
                        "entropy_score": 0.85,
                        "pruned": False,
                        "regulation": "eu_ai_act",
                    },
                    {
                        "id": "ai_act_def_3",
                        "label": "Definition (Art. 3)",
                        "node_type": "Definition",
                        "text_preview": "AI system: a machine-based system designed to operate with some degree...",
                        "entropy_score": 0.78,
                        "pruned": False,
                        "regulation": "eu_ai_act",
                    },
                    {
                        "id": "dsa_art_24",
                        "label": "Article 24 (DSA)",
                        "node_type": "Article",
                        "text_preview": "Systemic risks shall refer to risks that may have systemic implications...",
                        "entropy_score": 0.65,
                        "pruned": False,
                        "regulation": "dsa",
                    },
                ],
                "edges": [
                    {
                        "source": "ai_act_art_6",
                        "target": "ai_act_rec_47",
                        "relationship": ":INTERPRETS",
                        "strength": 0.95,
                    },
                    {
                        "source": "ai_act_art_6",
                        "target": "ai_act_def_3",
                        "relationship": ":USES_TERM",
                        "strength": 0.88,
                    },
                    {
                        "source": "ai_act_art_6",
                        "target": "dsa_art_24",
                        "relationship": ":OVERLAPS_WITH",
                        "strength": 0.72,
                    },
                ],
            },
            "citations": [
                "Article 6 (EU AI Act) - Classification rules for high-risk AI systems",
                "Recital 47 (EU AI Act) - Human oversight requirement",
                "Article 24 (DSA) - Systemic risks for VLOPs",
            ],
            "metrics": {
                "reasoning_steps": 3,
                "entropy_reduction_percent": 38,  # (0.18 + 0.08 + 0.12) * 100
                "tokens_saved": 1240,
                "latency_seconds": 2.0,
                "nodes_pruned": 0,
                "nodes_retrieved": 4,
            },
        }

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
