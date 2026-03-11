from typing import TypedDict, Optional, List, Dict, Any

class LegalNode(TypedDict):
    id: str         # e.g., "eu_ai_act_art_6"
    node_type: str  # "Article", "Recital", "Definition", "Paragraph"
    content: str    # Maps to 'full_text' or 'text'
    regulation: str # "eu_ai_act" or "dsa"
    metadata: Dict[str, Any] # {"chapter": "III", "section": "1", "score": 0.95}
    entropy_score: Optional[float]
    kl_score: Optional[float]  # <--- NEW: Added for Innovation 1 (KL Divergence)

class GraphState(TypedDict):
    query: str
    cypher_intent: Dict[str, Any]
    retrieved_nodes: List[LegalNode]
    links_found: List[Dict[str, Any]]
    pruned_context: List[LegalNode]
    reasoning_trace: List[str]
    final_answer: str
    hops: int
    is_accurate: bool
    # IMPORTANT: We keep 'metrics' here so the main_orchestrator can 
    # record the "Sustainability" results for your scientific paper.
    metrics: Dict[str, Any] 

mock_retrieved_nodes = [] # Kept for failover as requested