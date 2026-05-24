from sentence_transformers import SentenceTransformer, util

from engine_schema import GraphState

# Standardized model - keeping it lightweight for 1-core CPU
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- INNOVATION 1: SEMANTIC ENTROPY PRUNER (F1 & DIVERSITY OPTIMIZED) ---


def pruning_node(state: GraphState, threshold=0.38) -> GraphState:
    """
    Innovation 1: Semantic Entropy Pruning.
    Filters retrieved nodes based on their semantic relevance to the query.
    Updated: Added Redundancy Filter to prevent context clutter and ensure high F1.
    """
    print(f"--- SEMANTIC PRUNING (Threshold: {threshold}) ---")

    raw_nodes = state["retrieved_nodes"]
    if not raw_nodes:
        state["pruned_context"] = []
        return state

    # 1. Encode the Query
    query_embedding = model.encode(state["query"], convert_to_tensor=True)

    # Track all scores to handle the safety net
    scored_nodes = []
    for node in raw_nodes:
        node_embedding = model.encode(node["content"], convert_to_tensor=True)
        score = float(util.cos_sim(query_embedding, node_embedding))
        node["similarity_score"] = score
        scored_nodes.append(node)

    # 2. Sort nodes by relevance initially to process best nodes first
    scored_nodes = sorted(scored_nodes, key=lambda x: x["similarity_score"], reverse=True)

    # 3. Initial Pruning with REDUNDANCY FILTER
    # This prevents the LLM from getting 'distracted' by repetitive legal text.
    pruned_context = []
    for node in scored_nodes:
        if node["similarity_score"] >= threshold:
            # REDUNDANCY FILTER: If a node is 95% similar to one we already have, skip it.
            is_redundant = False
            node_emb = model.encode(node["content"], convert_to_tensor=True)

            for existing in pruned_context:
                existing_emb = model.encode(existing["content"], convert_to_tensor=True)
                sim = float(util.cos_sim(node_emb, existing_emb))
                if sim > 0.95:
                    is_redundant = True
                    print(
                        f"    ⏩ Skipping {node['id']} (Redundant with {existing['id']}: {sim:.4f})"
                    )
                    break

            if not is_redundant:
                pruned_context.append(node)

    # 4. --- TOP-3 SAFETY NET ---
    # Ensure density even if threshold was too strict, while maintaining the filter.
    if len(pruned_context) < 3 and len(scored_nodes) >= 3:
        print(f"⚠️ Context sparse ({len(pruned_context)} nodes). Filling to Top-3.")
        for node in scored_nodes:
            if len(pruned_context) >= 3:
                break
            if node not in pruned_context:
                pruned_context.append(node)
    elif not pruned_context and scored_nodes:
        pruned_context = scored_nodes[:1]  # Absolute fallback

    # --- NEW: GRAPH-AWARE STRUCTURAL LINKING ---
    # Ensure Article-Recital balance for the EMNLP 'Legal-Aware' argument.
    context_ids = [n["id"] for n in pruned_context]
    has_art = any("_art_" in cid for cid in context_ids)
    has_rec = any("_rec_" in cid for cid in context_ids)

    if has_art and not has_rec:
        recitals = [n for n in scored_nodes if "_rec_" in n["id"] and n not in pruned_context]
        if recitals:
            best_rec = sorted(recitals, key=lambda x: x["similarity_score"], reverse=True)[0]
            pruned_context.append(best_rec)
            print(f"🔗 Structural Link: Added Recital {best_rec['id']} to support Article context.")

    # 5. Final sorting for the Synthesizer
    pruned_context = sorted(pruned_context, key=lambda x: x["similarity_score"], reverse=True)

    # Log metrics for the paper
    for node in scored_nodes:
        status = "✅ Kept" if node in pruned_context else "🗑️ Pruned"
        print(f"    {status} {node['id']} (Score: {node['similarity_score']:.4f})")

    state["pruned_context"] = pruned_context
    return state
