from engine_schema import GraphState

# --- INNOVATION 2: AGENTIC SELF-CORRECTION (THE CRITIC) ---

def critic_node(state: GraphState) -> GraphState:
    """
    The 'Critic' Node that checks for legal completeness.
    Innovation 2: Structural Audit based on the KG Hierarchy.
    """
    print(f"--- CRITIC: Structural Audit (Hop {state['hops']}) ---")

    # 1. Gather all IDs in the current pruned context
    context_ids = [node["id"] for node in state["pruned_context"]]

    # 2. Join text for keyword-based backup check
    context_text = " ".join([node["content"] for node in state["pruned_context"]])

    missing_dependencies = []

    # RULE A: Hierarchical Link (From KG Plan Phase 4.2)
    # If Article 6 is present, the Critic checks for its interpretive Recital 53
    if "ai_act_art_6" in context_ids and "ai_act_rec_53" not in context_ids:
        missing_dependencies.append("ai_act_rec_53 (Interpretive Recital for Art 6)")

    # RULE B: Definitional Link
    # If 'High-risk' is mentioned but no Article 3 definitions exist
    if "High-risk" in context_text and "ai_act_art_3" not in str(context_ids):
        missing_dependencies.append("ai_act_art_3 (Definitions)")

    # --- Update State based on findings ---
    if missing_dependencies and state["hops"] < 3:
        print(f"Critic flagged missing links: {missing_dependencies}")
        state["is_accurate"] = False
        state["hops"] += 1
    else:
        if state["hops"] >= 3:
            print("Max hops reached. Proceeding with best available info.")
        else:
            print("Critic satisfied: Legal hierarchy is complete.")
        state["is_accurate"] = True

    return state

def self_correction_router(state: GraphState):
    """
    The 'Traffic Cop' (Conditional Edge) for LangGraph.
    Decides whether to generate the answer or re-traverse the Neo4j graph.
    """
    if state["is_accurate"]:
        return "generate_final_answer"
    else:
        return "re_traverse_graph"
