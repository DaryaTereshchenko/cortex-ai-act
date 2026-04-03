"""Configuration and shared types for the Judge RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


# ── Pipeline configuration ──────────────────────────────────────────────────


@dataclass
class JudgeRAGConfig:
    """All tuneable knobs for a single pipeline run."""

    # Neo4j connection
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"
    neo4j_database: str = "neo4j"

    # Ollama models (caller picks any model available on the local Ollama server)
    retrieval_model: str = "llama3.1:8b"
    generation_model: str = "llama3.1:8b"
    judge_model: str = "phi4:latest"

    # Ollama endpoint
    ollama_base_url: str = "http://localhost:11434"

    # Chunking
    chunk_size_tokens: int = 750  # target chunk size (500-1000 range)
    chunk_overlap_tokens: int = 125  # overlap (100-150 range)

    # Retrieval
    bm25_top_k: int = 25  # initial BM25 recall pool
    rerank_top_k: int = 10  # after semantic re-ranking
    adaptive_k_min: int = 3
    adaptive_k_max: int = 10
    similarity_threshold: float = 0.35

    # Embedding model for semantic re-ranking
    embedding_model: str = "all-MiniLM-L6-v2"

    # Query rewriting
    enable_query_rewriting: bool = True

    # Judge loop
    max_judge_attempts: int = 3
    judge_accept_threshold: float = 7.0  # score out of 10

    # Regulation filter
    regulation: str = "both"  # eu_ai_act | dsa | both


# ── Shared types ────────────────────────────────────────────────────────────


class Chunk(TypedDict):
    chunk_id: str
    source_id: str
    source_type: str  # Article, Paragraph, Recital, …
    regulation: str
    text: str
    metadata: dict[str, Any]


class RetrievedChunk(TypedDict):
    chunk: Chunk
    bm25_score: float
    semantic_score: float
    combined_score: float


class PipelineState(TypedDict):
    query: str
    rewritten_queries: list[str]
    chunks: list[Chunk]
    retrieved: list[RetrievedChunk]
    retrieval_similarity: float  # eval metric: avg cosine sim of retrieved vs query
    generated_answer: str
    judge_score: float
    judge_feedback: str
    attempt: int
    max_attempts: int
    is_accepted: bool
    uncertainty: str  # "low" | "medium" | "high"
    metrics: dict[str, Any]
