"""Knowledge-Graph service — FastAPI entry point."""

from __future__ import annotations

import os

from fastapi import FastAPI

app = FastAPI(
    title="CORTEX Knowledge-Graph Service",
    description="PDF → Markdown parsing, Neo4j schema management, and Cypher query API",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "knowledge-graph"}


@app.get("/")
async def root() -> dict:
    return {
        "service": "knowledge-graph",
        "environment": os.getenv("ENVIRONMENT", "development"),
    }
