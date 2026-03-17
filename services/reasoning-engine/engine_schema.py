from typing import Any, TypedDict

class LegalNode(TypedDict):
    id: str  
    node_type: str  
    content: str  
    regulation: str  
    metadata: dict[str, Any]  
    similarity_score: float | None 

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
    metrics: dict[str, Any]

mock_retrieved_nodes = []