import time
from datetime import datetime

import requests

from critic_engine import critic_node, self_correction_router
from engine_schema import GraphState, mock_retrieved_nodes
from pruning_engine import pruning_node
from synthesis_engine import (
    synthesis_node,
)


class KGConnector:
    BASE_URL = "http://localhost:8001/graph"

    @staticmethod
    def search_nodes(query: str) -> list[dict]:
        """Initial search: Finds the best starting points."""
        try:
            resp = requests.get(f"{KGConnector.BASE_URL}/search", params={"q": query, "limit": 5})
            if resp.status_code == 200:
                nodes = []
                for r in resp.json():
                    node_data = r["result"]
                    nodes.append(
                        {
                            "id": node_data["id"],
                            "node_type": r["label"],
                            "content": node_data.get("full_text") or node_data.get("text") or "",
                            "regulation": node_data.get("regulation"),
                            "metadata": {"score": r["score"]},
                            "similarity_score": None,
                        }
                    )
                return nodes
        except Exception as e:
            print(f"Connection to KG failed: {e}")
        return mock_retrieved_nodes

    @staticmethod
    def get_neighbors(node_id: str) -> list[dict]:
        """RE-TRAVERSAL: Grabs connected nodes (Recitals, Definitions, etc.)."""
        try:
            resp = requests.get(f"{KGConnector.BASE_URL}/traverse/{node_id}", params={"depth": 1})
            if resp.status_code == 200:
                data = resp.json()
                neighbors = []
                for n in data.get("neighbors", []):
                    # Grab full_text/text instead of just the title to ensure factual accuracy
                    neighbors.append(
                        {
                            "id": n["id"],
                            "node_type": n["label"],
                            "content": n.get("full_text")
                            or n.get("text")
                            or n.get("title")
                            or n.get("id"),
                            "regulation": n.get("id", "").split("_")[0],
                            "metadata": {"parent": node_id},
                            "similarity_score": None,
                        }
                    )
                return neighbors
        except Exception as e:
            print(f"Traversal failed: {e}")
        return []


def run_cortex_engine(
    user_query: str,
    *,
    max_hops: int = 2,
    enable_pruning: bool = True,
    enable_self_correction: bool = True,
    pruning_threshold: float = 0.45,
):
    start_time = time.time()

    # Performance Trackers
    global_raw_chars = 0
    global_kept_chars = 0

    state: GraphState = {
        "query": user_query,
        "cypher_intent": {},
        "retrieved_nodes": KGConnector.search_nodes(user_query),
        "links_found": [],
        "pruned_context": [],
        "reasoning_trace": [f"🚀 Query received: {user_query}"],
        "final_answer": "",
        "hops": 0,
        "is_accurate": False,
        "metrics": {
            "tokens_saved": 0,
            "optimization_ratio": 0.0,
            "nodes_pruned": 0,
            "latency_seconds": 0.0,
        },
    }

    # --- AGENTIC LOOP ---
    while state["hops"] < max_hops:
        # 1. Capture Encountered volume
        hop_raw_chars = sum(len(n.get("content", "")) for n in state["retrieved_nodes"])
        global_raw_chars += hop_raw_chars
        initial_node_count = len(state["retrieved_nodes"])

        # 2. Semantic Pruning Layer
        if enable_pruning:
            state = pruning_node(state, threshold=pruning_threshold)
        else:
            # Ablation mode: bypass semantic pruning and keep all retrieved context.
            state["pruned_context"] = list(state["retrieved_nodes"])
            state["reasoning_trace"].append("Pruner disabled: retaining full retrieved context.")

        # 3. Capture Accepted volume
        hop_kept_chars = sum(len(n.get("content", "")) for n in state["pruned_context"])
        global_kept_chars += hop_kept_chars

        # 4. Update Cumulative Metrics
        saved = max(0, (hop_raw_chars - hop_kept_chars) // 4)
        state["metrics"]["tokens_saved"] += saved
        state["metrics"]["nodes_pruned"] += max(
            0, initial_node_count - len(state["pruned_context"])
        )

        # 5. Semantic Critic Layer
        if enable_self_correction:
            state = critic_node(state)
        else:
            state["is_accurate"] = True
            state["reasoning_trace"].append("Critic disabled: skipping self-correction checks.")

        if self_correction_router(state) == "generate_final_answer":
            break

        if not state["pruned_context"]:
            state["reasoning_trace"].append("⚠️ Logic Halt: No context remained.")
            break

        # 6. Re-traversal
        target_node = state["pruned_context"][0]["id"]
        state["reasoning_trace"].append(f"🔄 Re-traversing graph from {target_node}...")

        new_nodes = KGConnector.get_neighbors(target_node)
        state["retrieved_nodes"].extend(new_nodes)
        if enable_self_correction:
            state["hops"] += 1
        else:
            # With critic disabled we do not loop for corrective traversal.
            break

    # --- FINAL METRICS CALCULATION ---
    # Calculating the reduction ratio
    final_reduction = 0.0
    if global_raw_chars > 0:
        final_reduction = float(round((global_raw_chars - global_kept_chars) / global_raw_chars, 3))

    final_reduction = max(0.0, min(1.0, final_reduction))
    state["metrics"]["optimization_ratio"] = final_reduction
    state["metrics"]["latency_seconds"] = round(time.time() - start_time, 2)

    # --- SYNTHESIS LAYER ---
    state = synthesis_node(state)

    # --- SCHEMA HANDSHAKE FORMATTING ---
    formatted_steps = []
    for i, step in enumerate(state["reasoning_trace"]):
        # Map the internal trace message to a Web-UI recognized Agent role
        agent_name = "Retriever" # Default
        if any(keyword in step for keyword in ["Critic", "Validation", "missing context"]):
            agent_name = "Critic"
        elif any(keyword in step for keyword in ["SYNTHESIZING", "QWEN", "GENERATING"]):
            agent_name = "Synthesizer"

        formatted_steps.append(
            {
                "step_number": int(i + 1),
                "agent": agent_name,
                "action": str(step),
                "retrieved_nodes": [str(n["id"]) for n in state["pruned_context"]],
                "entropy_reduction": 0.0,
                "timestamp": datetime.now().isoformat(),
            }
        )

    
    # Return structure mapped for Gateway (Handshake Sync with WebUI Schema)
    return {
        "query_id": f"query_{int(time.time())}",
        "status": "completed",
        "question": user_query,
        "final_answer": str(state["final_answer"]),
        "reasoning_steps": formatted_steps,
        "graph_data": {"nodes": [], "edges": []},
        "citations": [f"{n['id']} ({n['regulation'].upper()})" for n in state["pruned_context"]],
        "metrics": {
            "reasoning_steps": len(formatted_steps),
            "tokens_saved": int(state["metrics"]["tokens_saved"]),
            "nodes_pruned": int(state["metrics"]["nodes_pruned"]),
            "latency_seconds": float(state["metrics"]["latency_seconds"]),
            "entropy_reduction": float(final_reduction),
            "enable_pruning": bool(enable_pruning),
            "enable_self_correction": bool(enable_self_correction),
            "pruning_threshold": float(pruning_threshold),
        },
    }


if __name__ == "__main__":
    res = run_cortex_engine("Final System Test with Synthesis")
    print(f"✅ Final Result Metrics: {res['metrics']}")
    print(f"✅ Final Answer Snippet: {res['final_answer'][:100]}...")
