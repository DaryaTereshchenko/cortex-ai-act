"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """User's legal query request."""

    question: str = Field(
        ..., min_length=10, max_length=2000, description="Legal compliance question"
    )
    regulation: str = Field(
        default="eu_ai_act",
        pattern="^(eu_ai_act|dsa|both)$",
        description="Target regulation scope",
    )
    max_hops: int = Field(default=3, ge=1, le=5, description="Max graph traversal depth")
    enable_pruning: bool = Field(default=True, description="Enable semantic entropy pruning")
    enable_self_correction: bool = Field(default=True, description="Enable agentic self-correction")
    pruning_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Semantic pruning threshold (higher = stricter pruning)",
    )


class LegalNode(BaseModel):
    """Legal document node from knowledge graph (matches KG service schema)."""

    id: str = Field(..., description="Unique node identifier (e.g., 'ai_act_art_6')")
    node_type: str = Field(..., description="Node type: Article, Recital, Definition, TextChunk")
    content: str = Field(..., description="Full text of the legal article/definition")
    regulation: str = Field(..., description="Source regulation: eu_ai_act or dsa")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (chapter, section, url)"
    )
    entropy_score: float | None = Field(default=None, description="Semantic entropy score")


class GraphNode(BaseModel):
    """Node in the reasoning graph (for visualization)."""

    id: str = Field(..., description="Unique node identifier")
    label: str = Field(..., description="Display label (e.g., 'Article 6')")
    node_type: str = Field(..., description="Node type: Article, Recital, Definition, etc.")
    text_preview: str | None = Field(default=None, description="First 200 chars of text")
    entropy_score: float | None = Field(default=None, description="Semantic entropy score")
    pruned: bool = Field(default=False, description="Whether node was pruned")
    regulation: str = Field(..., description="Source regulation")


class GraphEdge(BaseModel):
    """Edge between graph nodes."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relationship: str = Field(..., description="Relationship type")
    strength: float = Field(default=1.0, description="Edge weight for visualization")


class ReasoningStep(BaseModel):
    """Single step in the agentic reasoning loop."""

    step_number: int = Field(..., description="Step sequence number")
    agent: str = Field(..., description="Agent performing action: Retriever, Critic, Synthesizer")
    action: str = Field(..., description="Action description")
    retrieved_nodes: list[str] = Field(
        default_factory=list, description="Nodes retrieved in this step"
    )
    entropy_reduction: float | None = Field(
        default=None, description="Percentage of context reduced by pruning"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Step timestamp")


class QueryResponse(BaseModel):
    """Complete response to a query."""

    query_id: str = Field(..., description="Unique query identifier")
    status: str = Field(..., description="Query status: processing, completed, failed")
    question: str = Field(..., description="Original user question")
    final_answer: str | None = Field(default=None, description="Generated answer")
    reasoning_steps: list[ReasoningStep] = Field(
        default_factory=list, description="Reasoning trace"
    )
    graph_data: dict[str, Any] = Field(default_factory=dict, description="Graph nodes and edges")
    citations: list[str] = Field(default_factory=list, description="Article citations")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    error: str | None = Field(default=None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Service health status response."""

    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    service: str = Field(default="web-ui", description="Service name")
    reasoning_engine_available: bool = Field(..., description="Reasoning engine connectivity")
    knowledge_graph_available: bool = Field(..., description="Knowledge graph connectivity")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check time")
