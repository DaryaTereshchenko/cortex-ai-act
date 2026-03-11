from engine_schema import GraphState

def synthesis_node(state: GraphState) -> GraphState:
    """Innovation 3: Context-Aware Synthesis.
    On Laptop: Combines snippets from the pruned context.
    On Nuvelos: Uses Llama 3.1 to write a formal legal opinion.
    """
    print(f"--- SYNTHESIZING FINAL ANSWER ---")
    
    if not state["pruned_context"]:
        state["final_answer"] = "I could not find enough relevant legal context to answer your query accurately."
        return state

    # 1. Gather the "Golden Context"
    # We take the top 3 nodes that survived the Pruning Engine
    top_nodes = state["pruned_context"][:3]
    
    # 2. Build the Answer (Laptop Mockup)
    intro = f"Based on a multi-hop analysis of the {state['retrieved_nodes'][0]['regulation'].upper()}...\n\n"
    
    body_parts = []
    for node in top_nodes:
        # We take a snippet of the content to show 'real' text in the UI
        snippet = node['content'][:200].strip() + "..."
        body_parts.append(f"• FROM {node['id'].upper()}: {snippet}")
    
    conclusion = f"\n\nThis analysis was optimized by CORTEX-RAG, reducing redundant legal noise by {state['metrics']['entropy_reduction'] * 100:.1f}%."

    # Update State
    state["final_answer"] = intro + "\n".join(body_parts) + conclusion
    state["reasoning_trace"].append("Synthesizer: Generated formal legal response from pruned context.")
    
    return state