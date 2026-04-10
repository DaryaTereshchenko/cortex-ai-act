from sentence_transformers import SentenceTransformer, util

from engine_schema import GraphState

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Lazy-loaded model cache keyed by model name
_models: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]

# --- INNOVATION 1: SEMANTIC ENTROPY PRUNER ---


def pruning_node(state: GraphState, threshold=0.45, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> GraphState:
    """
    Innovation 1: Semantic Entropy Pruning.
    Filters retrieved nodes based on their semantic relevance to the query.
    Ensures the synthesizer only receives contextually relevant legal data.
    """
    print(f"--- SEMANTIC PRUNING (Threshold: {threshold}) ---")

    raw_nodes = state["retrieved_nodes"]
    if not raw_nodes:
        return state

    model = _get_model(embedding_model)
    query_embedding = model.encode(state["query"], convert_to_tensor=True)
    pruned_context = []

    for node in raw_nodes:
        node_embedding = model.encode(node["content"], convert_to_tensor=True)
        score = float(util.cos_sim(query_embedding, node_embedding))

        if score >= threshold:
            node["similarity_score"] = score
            pruned_context.append(node)
        else:
            print(f"    🗑️ Pruned {node['id']} (Low Relevance: {score:.4f})")

    # Sort by relevance to assist the synthesizer
    pruned_context = sorted(pruned_context, key=lambda x: x["similarity_score"], reverse=True)

    state["pruned_context"] = pruned_context
    return state
