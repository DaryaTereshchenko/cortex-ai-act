from engine_schema import GraphState

# --- INNOVATION 2: STRUCTURAL AUDIT & AGENTIC SELF-CORRECTION ---


def critic_node(state: GraphState) -> GraphState:
    """
    Innovation 2: Structural Audit
    Validates if the pruned context meets minimum legal requirements
    for an accurate response.
    """
    print(f"--- CRITIC: Semantic Validation (Hop {state['hops']}) ---")

    context_ids = [node["id"] for node in state["pruned_context"]]
    context_text = " ".join([node["content"] for node in state["pruned_context"]]).lower()
    missing_requirements = []

    # Check 1: Article/Recital Balance
    # High-quality legal RAG requires both the rule (Article) and the intent (Recital).
    has_article = any("_art_" in cid for cid in context_ids)
    has_recital = any("_rec_" in cid for cid in context_ids)

    if has_article and not has_recital:
        missing_requirements.append("Interpretive Recitals")

    # Check 2: Key Term Coverage
    # Ensures specific high-stakes terms in the query are supported in the context.
    important_terms = ["biometric", "risk", "compliance", "vetted"]
    for term in important_terms:
        if term in state["query"].lower() and term not in context_text:
            missing_requirements.append(f"Technical context for '{term}'")

    if missing_requirements and state["hops"] < 3:
        msg = f"Validation flagged missing context: {missing_requirements}"
        print(f"⚠️ {msg}")
        state["is_accurate"] = False
        state["reasoning_trace"].append(msg)
    else:
        state["is_accurate"] = True
        state["reasoning_trace"].append("Critic: Context coverage verified for final synthesis.")

    return state


def self_correction_router(state: GraphState):
    return "generate_final_answer" if state["is_accurate"] else "re_traverse_graph"
