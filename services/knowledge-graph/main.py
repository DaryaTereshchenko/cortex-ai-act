"""Knowledge-Graph service — FastAPI entry point."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import init_connection
from api.routes import router as graph_router
from graph.connection import Neo4jConnection

log = logging.getLogger(__name__)

_neo4j: Neo4jConnection | None = None

_MAX_RETRIES = 10
_RETRY_DELAY = 3  # seconds


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Connect to Neo4j on startup (with retries), close on shutdown."""
    global _neo4j
    _neo4j = Neo4jConnection()
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            _neo4j.connect()
            init_connection(_neo4j)
            log.info("Neo4j connection ready (attempt %d)", attempt)
            break
        except Exception as exc:
            if attempt < _MAX_RETRIES:
                log.warning(
                    "Neo4j not ready (attempt %d/%d): %s — retrying in %ds",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    _RETRY_DELAY,
                )
                _neo4j._driver = None  # reset so connect() retries
                await asyncio.sleep(_RETRY_DELAY)
            else:
                log.error(
                    "Neo4j not available after %d attempts — graph endpoints will return 503",
                    _MAX_RETRIES,
                )
    yield
    if _neo4j:
        _neo4j.close()


app = FastAPI(
    title="CORTEX Knowledge-Graph Service",
    description="Regulation knowledge graph: Neo4j ingestion, Cypher query API, and full-text search",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(graph_router)


@app.get("/health")
async def health() -> dict:
    neo4j_ok = _neo4j.is_healthy() if _neo4j else False
    return {
        "status": "ok" if neo4j_ok else "degraded",
        "service": "knowledge-graph",
        "neo4j": "connected" if neo4j_ok else "unavailable",
    }


@app.get("/")
async def root() -> dict:
    return {
        "service": "knowledge-graph",
        "version": "0.2.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "endpoints": [
            "/health",
            "/graph/ingest",
            "/graph/stats",
            "/graph/regulations",
            "/graph/article/{regulation}/{number}",
            "/graph/chapter/{regulation}/{number}",
            "/graph/recitals/{regulation}",
            "/graph/annexes/{regulation}",
            "/graph/definitions/{regulation}",
            "/graph/search?q=...",
            "/graph/traverse/{node_id}",
            "/graph/cypher",
        ],
    }
