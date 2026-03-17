from engine_schema import GraphState

# --- INNOVATION 3: CONTEXT-AWARE SYNTHESIS ---

def synthesis_node(state: GraphState) -> GraphState:
    """
    Innovation 3: Context-Aware Synthesis:
    Integrates the optimized semantic context into a structured legal response.
    """
    print("--- SYNTHESIZING FINAL ANSWER ---")

    if not state["pruned_context"]:
        state["final_answer"] = "No relevant legal context found to support this query."
        return state

    top_nodes = state["pruned_context"][:3]
    intro = f"Analysis based on optimized {state['retrieved_nodes'][0]['regulation'].upper()} context:\n"
    
    body_parts = []
    for node in top_nodes:
        snippet = node["content"][:1000].strip() + "..."
        body_parts.append(f"• SOURCE {node['id'].upper()}: {snippet}")

    # Report optimization based on character count reduction
    optimization_val = state["metrics"].get("entropy_reduction", 0.0) * 100
    conclusion = f"\n\nContext optimized for high-fidelity response. (Efficiency Gain: {optimization_val:.1f}%)"

    state["final_answer"] = intro + "\n".join(body_parts) + conclusion
    state["reasoning_trace"].append("Synthesizer: Response generated from semantically optimized context.")
    return state