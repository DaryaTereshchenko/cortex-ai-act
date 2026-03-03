from typing import Dict
from engine_schema import GraphState, mock_retrieved_nodes
from pruning_engine import pruning_node
from critic_engine import self_correction_router
import json

# --- Updated Pre-Retrieval (The 'Cypher' Simulator) ---
def pre_retrieval_optimizer(query: str) -> Dict:
    """
    PRE-RETRIEVAL: Now generates a 'Search Strategy' instead of just a string.
    This tells the KG Lead what logic we want to traverse.
    """
    print(f"--- PRE-RETRIEVAL: Generating Cypher Logic ---")
    return {
        "original_query": query,
        "cypher_intent": "MATCH (a:Article)-[:USES_TERM]->(d:Definition) WHERE a.full_text CONTAINS 'high-risk' RETURN a, d",
        "depth": 2
    }

# --- Updated Post-Retrieval Reranker ---
def post_retrieval_reranker(nodes: list) -> list:
    print(f"--- POST-RETRIEVAL: Re-ranking {len(nodes)} nodes ---")
    return sorted(nodes, key=lambda x: x.get("entropy_score", 0))

# --- Updated Critic (The 'Structural' Audit) ---
def critic_node(state: GraphState) -> GraphState:
    """
    Innovation 2: Audits the hierarchy based on the KG Plan (Phase 4.2).
    Checks for interpretive links (Recitals) and definitional links.
    """
    print(f"--- CRITIC: Structural Audit (Hop {state['hops']}) ---")
    context_ids = [n["id"] for n in state["retrieved_nodes"]]
    
    # Check 1: Article-to-Recital Interpretive Gap
    # If we have Art 6, we MUST have Recital 53 to understand 'intent'
    if "ai_act_art_6" in context_ids and "ai_act_rec_53" not in context_ids:
        print("Critic Alert: Missing interpretive Recital 53 for Article 6.")
        state["is_accurate"] = False
        state["hops"] += 1
    else:
        state["is_accurate"] = True
    return state

# --- FINALIZED REASONING ENGINE ---

def run_cortex_engine(user_query: str):
    # 1. PRE-RETRIEVAL Optimization
    search_strategy = pre_retrieval_optimizer(user_query)
    
    state: GraphState = {
        "query": user_query,
        "cypher_intent": search_strategy,
        "retrieved_nodes": mock_retrieved_nodes.copy(),
        "links_found": [],
        "pruned_context": [],
        "final_answer": "",
        "hops": 0,
        "is_accurate": False
    }

    print(f"🚀 CORTEX-RAG v2: Processing '{user_query}'")

    while state["hops"] < 3:
        # 2. PRUNING (Innovation 1)
        # We now prune nodes from 'retrieved_nodes'
        state = pruning_node(state, model=None, tokenizer=None)

        # 3. RE-RANKING
        state["pruned_context"] = post_retrieval_reranker(state["pruned_context"])

        # 4. CRITIC (Structural Audit)
        state = critic_node(state)

        next_step = self_correction_router(state)
        
        if next_step == "generate_final_answer":
            break
        else:
            print("🔄 Re-traversing Graph for missing hierarchy...")
            # Simulate KG Lead's :INTERPRETS relationship
            state["retrieved_nodes"].append({
                "id": "ai_act_rec_53",
                "node_type": "Recital",
                "content": "Recital 53: Explains legislative intent for high-risk rules...",
                "regulation": "eu_ai_act",
                "metadata": {"interprets": "ai_act_art_6"},
                "entropy_score": None
            })

    # 5. FINAL SYNTHESIS
    state["final_answer"] = f"Final answer based on Article 6 and its interpretive Recital 53..."

    # --- API CONTRACT OUTPUT ---
    initial_count = len(state["retrieved_nodes"])
    final_count = len(state["pruned_context"])
    reduction_pct = ((initial_count - final_count) / initial_count) * 100

    api_output = {
        "final_answer": state["final_answer"],
        "citations": [n["id"] for n in state["pruned_context"]],
        "metrics": {
            "total_hops": state["hops"],
            "entropy_pruning_pct": f"{reduction_pct:.2f}%",
            "cross_regulatory_active": any(n["regulation"] == "dsa" for n in state["pruned_context"])
        }
    }

    print(f"✅ ENGINE FINISHED. Sustainability: {api_output['metrics']['entropy_pruning_pct']} saved.")
    return api_output

if __name__ == "__main__":
    run_cortex_engine("What are the rules for high-risk systems?")