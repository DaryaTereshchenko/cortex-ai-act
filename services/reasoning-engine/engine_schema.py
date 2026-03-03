from typing import TypedDict


# Matches the KG Lead's Node Properties (Regulation, Chapter, Article, etc.)
class LegalNode(TypedDict):
    id: str                # e.g., "ai_act_art_6"
    node_type: str         # "Article", "Recital", "Definition", "TextChunk"
    content: str           # The 'full_text' or 'definition_text'
    regulation: str        # "eu_ai_act" or "dsa"
    metadata: dict         # {"chapter": "III", "section": "1", "url": "..."}
    entropy_score: float | None

class GraphState(TypedDict):
    query: str                   # User's natural language question
    cypher_intent: dict          # The 'Search Strategy' for the Graph Lead
    retrieved_nodes: list[LegalNode]
    links_found: list[dict]      # Relationships like :INTERPRETS or :OVERLAPS_WITH
    pruned_context: list[LegalNode]
    final_answer: str
    hops: int
    is_accurate: bool

# --- ENRICHED MOCK DATA ---
# This simulates the hierarchical nodes your colleague will provide
mock_retrieved_nodes = [
    {
        "id": "ai_act_art_6",
        "node_type": "Article",
        "content": "Classification rules for high-risk AI systems...",
        "regulation": "eu_ai_act",
        "metadata": {"chapter": "III"},
        "entropy_score": None
    },
    {
        "id": "ai_act_art_3_1",
        "node_type": "Definition",
        "content": "AI system means a machine-based system...",
        "regulation": "eu_ai_act",
        "metadata": {"source": "Article 3"},
        "entropy_score": None
    },
    {
        "id": "dsa_art_34",
        "node_type": "Article",
        "content": "Providers of VLOPs shall identify systemic risks...",
        "regulation": "dsa",
        "metadata": {"chapter": "V"},
        "entropy_score": None
    }
]

print("Step 1 (v2) Complete: Enriched Hierarchical Schema Defined.")
