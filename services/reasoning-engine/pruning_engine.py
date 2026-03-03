import random

import torch
import torch.nn.functional as fn

# CHANGE: Import LegalNode instead of LegalDocument to match v2 Schema
from engine_schema import GraphState

# --- INNOVATION 1: SEMANTIC ENTROPY PRUNER (v2) ---


def calculate_pruning_entropy(text, model, tokenizer):
    """
    Innovation 1 Logic.
    If no GPU/Model is present (Laptop), it returns a random mock score.
    If Llama 3.1 is present (Nuvelos), it runs the real Negative Log-Likelihood math.
    """
    # LOCAL MOCK MODE: For your laptop development
    if model is None or tokenizer is None:
        # We generate a random entropy score between 0.1 and 0.8
        return random.uniform(0.1, 0.8)

    # NUVELOS GPU MODE: Real math execution
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # Get model predictions (logits)
        outputs = model(**inputs)
        logits = outputs.logits

        # Convert to probabilities
        probs = fn.softmax(logits, dim=-1)

        # Calculate Entropy: H = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        # Average across all tokens in the legal segment
        return torch.mean(entropy).item()


def pruning_node(state: GraphState, model, tokenizer, threshold=0.35) -> GraphState:
    """
    The LangGraph Node that prunes the context.
    Threshold: lower p = more aggressive pruning (Innovation 1).
    Now operates on 'retrieved_nodes' instead of docs.
    """
    print(f"--- PRUNING CONTEXT (Threshold: {threshold}) ---")

    # CHANGE: Use 'retrieved_nodes' from the updated State
    raw_nodes = state["retrieved_nodes"]
    scored_nodes = []

    # 1. Score each LegalNode based on entropy (Information Density)
    for node in raw_nodes:
        # We use the 'content' field which contains the legal text
        score = calculate_pruning_entropy(node["content"], model, tokenizer)
        node["entropy_score"] = score
        scored_nodes.append(node)

    # 2. Percentile-Based Filtering
    # Lower entropy = Higher certainty/signal. We keep the most informative nodes.
    scored_nodes.sort(key=lambda x: x["entropy_score"])

    # Simple threshold logic: keep the bottom (1 - threshold)% of entropy scores
    cutoff = max(1, int(len(scored_nodes) * (1 - threshold)))
    pruned_context = scored_nodes[:cutoff]

    print(f"Pruned {len(raw_nodes) - len(pruned_context)} redundant nodes.")

    # CHANGE: Update 'pruned_context' in the state
    state["pruned_context"] = pruned_context
    return state
