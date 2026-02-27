"""Benchmarking service — FastAPI entry point."""

from __future__ import annotations

import os

from fastapi import FastAPI

app = FastAPI(
    title="CORTEX Benchmarking Service",
    description="Token-reduction, inference-latency, and sustainability metrics",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "benchmarking"}


@app.get("/")
async def root() -> dict:
    return {
        "service": "benchmarking",
        "environment": os.getenv("ENVIRONMENT", "development"),
    }
