"""Reasoning-Engine service — FastAPI entry point."""

from __future__ import annotations

import os

from fastapi import FastAPI

app = FastAPI(
    title="CORTEX Reasoning Engine",
    description="LLM-powered reasoning with Entropy Pruning and Self-Correction Loop",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "reasoning-engine"}


@app.get("/")
async def root() -> dict:
    return {
        "service": "reasoning-engine",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "model_id": os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    }
