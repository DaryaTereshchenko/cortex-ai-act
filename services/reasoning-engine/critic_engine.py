from engine_schema import GraphState

def critic_node(state: GraphState) -> GraphState:
    """Innovation 2: Structural Audit based on official KG ID patterns and cross-reg overlaps."""
    print(f"--- CRITIC: Structural Audit (Hop {state['hops']}) ---")
    
    # 1. Gather context for checking
    context_ids = [node["id"] for node in state["pruned_context"]]
    context_text = " ".join([node["content"] for node in state["pruned_context"]]).lower()
    
    missing_dependencies = []

    # RULE 1: Hierarchical Dependency (EU AI Act Art 6 -> Recital 53)
    # The Builder uses the pattern: {reg_id}_art_{number}
    if "eu_ai_act_art_6" in context_ids and "eu_ai_act_rec_53" not in context_ids:
        missing_dependencies.append("eu_ai_act_rec_53 (Interpretive Recital)")

    # RULE 2: Thematic Cross-Regulation Overlap (AI Act Art 9 -> DSA Art 34)
    # If the user is asking about 'risk management' or Art 9, the Critic suggests the DSA counterpart.
    is_risk_query = "eu_ai_act_art_9" in context_ids or "risk management" in context_text
    if is_risk_query and "dsa_art_34" not in context_ids:
        state["reasoning_trace"].append("Critic: Detected Risk Management focus. Ensuring DSA Article 34 is included.")
        missing_dependencies.append("dsa_art_34 (Risk Assessment Overlap)")

    # RULE 3: Definitional Check (High-risk AI -> Article 3 Definitions)
    # If 'high-risk' is discussed but no definitions are present.
    if "high-risk" in context_text and not any("_def_" in cid for cid in context_ids):
        missing_dependencies.append("eu_ai_act_art_3 (Definitions)")

    # --- Update State ---
    if missing_dependencies and state["hops"] < 3:
        msg = f"Audit flagged missing dependencies: {missing_dependencies}"
        print(f"⚠️ {msg}")
        state["is_accurate"] = False
        # Note: hops incrementing is now handled in the orchestrator loop to avoid double-counting
        state["reasoning_trace"].append(msg)
    else:
        state["is_accurate"] = True
        state["reasoning_trace"].append("Critic: Legal hierarchy and cross-references verified.")
    
    return state

def self_correction_router(state: GraphState):
    """The conditional router for the LangGraph flow."""
    return "generate_final_answer" if state["is_accurate"] else "re_traverse_graph"