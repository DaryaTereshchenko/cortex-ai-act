from typing import Any, TypedDict


class LegalNode(TypedDict):
    id: str  # e.g., "eu_ai_act_art_6"
    node_type: str  # "Article", "Recital", "Definition", "Paragraph"
    content: str  # Maps to 'full_text' or 'text'
    regulation: str  # "eu_ai_act" or "dsa"
    metadata: dict[str, Any]  # {"chapter": "III", "section": "1", "score": 0.95}
    entropy_score: float | None
    kl_score: float | None  # <--- NEW: Added for Innovation 1 (KL Divergence)


class GraphState(TypedDict):
    query: str
    cypher_intent: dict[str, Any]
    retrieved_nodes: list[LegalNode]
    links_found: list[dict[str, Any]]
    pruned_context: list[LegalNode]
    reasoning_trace: list[str]
    final_answer: str
    hops: int
    is_accurate: bool
    # IMPORTANT: We keep 'metrics' here so the main_orchestrator can
    # record the "Sustainability" results for your scientific paper.
    metrics: dict[str, Any]


mock_retrieved_nodes = []  # Kept for failover as requested
