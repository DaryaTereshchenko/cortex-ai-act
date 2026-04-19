import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

from engine_schema import GraphState

# Ensure baselines package is importable
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.model_registry import get_model as _registry_get_model  # noqa: E402

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _get_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return _registry_get_model(model_name)


# --- INNOVATION 1: SEMANTIC ENTROPY PRUNER ---


def pruning_node(
    state: GraphState, threshold=0.45, embedding_model: str = DEFAULT_EMBEDDING_MODEL
) -> GraphState:
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
