"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """User's legal query request."""

    question: str = Field(..., min_length=10, max_length=2000, description="Legal compliance question")
    regulation: str = Field(
        default="eu_ai_act",
        pattern="^(eu_ai_act|dsa|both)$",
        description="Target regulation scope"
    )
    max_hops: int = Field(default=3, ge=1, le=5, description="Max graph traversal depth")
    enable_pruning: bool = Field(default=True, description="Enable semantic entropy pruning")
    enable_self_correction: bool = Field(default=True, description="Enable agentic self-correction")


class GraphNode(BaseModel):
    """Node in the reasoning graph."""

    id: str = Field(..., description="Unique node identifier")
    label: str = Field(..., description="Display label (e.g., 'Article 6')")
    node_type: str = Field(..., description="Node type: Article, Recital, Definition, etc.")
    text_preview: Optional[str] = Field(default=None, description="First 200 chars of text")
    entropy_score: Optional[float] = Field(default=None, description="Semantic entropy score")
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
    retrieved_nodes: List[str] = Field(default_factory=list, description="Nodes retrieved in this step")
    entropy_reduction: Optional[float] = Field(
        default=None,
        description="Percentage of context reduced by pruning"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Step timestamp")


class QueryResponse(BaseModel):
    """Complete response to a query."""

    query_id: str = Field(..., description="Unique query identifier")
    status: str = Field(..., description="Query status: processing, completed, failed")
    question: str = Field(..., description="Original user question")
    final_answer: Optional[str] = Field(default=None, description="Generated answer")
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list, description="Reasoning trace")
    graph_data: Dict[str, Any] = Field(default_factory=dict, description="Graph nodes and edges")
    citations: List[str] = Field(default_factory=list, description="Article citations")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Service health status response."""

    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    service: str = Field(default="web-ui", description="Service name")
    reasoning_engine_available: bool = Field(..., description="Reasoning engine connectivity")
    knowledge_graph_available: bool = Field(..., description="Knowledge graph connectivity")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check time")
