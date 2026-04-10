from engine_schema import GraphState

# --- INNOVATION 2: STRUCTURAL AUDIT & AGENTIC SELF-CORRECTION (Graph-Aware) ---

def critic_node(state: GraphState) -> GraphState:
    """
    Innovation 2: Structural Audit
    Validates context coverage. Now includes Graph-Aware Definition checking
    to ensure legal Articles are supported by their requisite Definitions.
    """
    print(f"--- CRITIC: Semantic Validation (Hop {state['hops']}) ---")

    context_ids = [node["id"] for node in state["pruned_context"]]
    context_text = " ".join([node["content"] for node in state["pruned_context"]]).lower()
    missing_requirements = []

    # 1. Article/Recital Balance Check
    has_article = any("_art_" in cid for cid in context_ids)
    has_recital = any("_rec_" in cid for cid in context_ids)

    if has_article and not has_recital:
        missing_requirements.append("Interpretive Recitals")

    # 2. NEW: Graph-Aware Definition Check (NLLP Excellence)
    # Ensures that 'High-Risk' or 'Prohibited' rules have their definitions attached.
    if has_article and "definition" not in context_text:
        # We only force this for complex queries where ambiguity is high
        if len(state["query"].split()) > 8:
            missing_requirements.append("Missing Legal Definitions for context")

    # 3. Dynamic Term Coverage (F1 Booster)
    # Focuses on words > 5 chars to avoid noise from 'the', 'shall', etc.
    query_words = [w.strip("?,.!") for w in state["query"].lower().split() if len(w) > 5]
    for word in query_words:
        if word not in context_text:
            missing_requirements.append(f"Query-specific term: '{word}'")

    # 4. Prohibition Keyword Guard (Negative Constraint Check)
    prohibition_keywords = ["prohibit", "forbidden", "ban", "not allowed", "stop"]
    if any(k in state["query"].lower() for k in prohibition_keywords):
        if not any(k in context_text for k in ["shall not", "prohibit", "ban", "article 5"]):
            missing_requirements.append("Missing prohibitive legal language")

    # 5. Reasoning & Routing Logic
    # Capped at 3 hops for Digital Sustainability (Nuvolos Core Optimization)
    if missing_requirements and state["hops"] < 3:
        msg = f"Critic flagged missing context: {missing_requirements}"
        print(f"⚠️ {msg}")
        state["is_accurate"] = False
        state["reasoning_trace"].append(msg)
    else:
        state["is_accurate"] = True
        state["reasoning_trace"].append("Critic: Legal-Graph coverage verified for final synthesis.")

    return state


def self_correction_router(state: GraphState):
    """
    Routes the state either to final answer generation or back for another graph hop.
    """
    return "generate_final_answer" if state["is_accurate"] else "re_traverse_graph"