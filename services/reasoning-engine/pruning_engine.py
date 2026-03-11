import random

import torch
import torch.nn.functional as fn

from engine_schema import GraphState

# --- INNOVATION 1: KL DIVERGENCE + ENTROPY PRUNER ---


def calculate_kl_divergence(p_text, q_text, model, tokenizer):
    """
    Innovation 1: Information Gain logic.
    Calculates how much 'new' info q_text adds to p_text.
    """
    if model is None or tokenizer is None:
        # MOCK MODE: Returns a score 0-1.
        # Low = Q adds nothing new to P. High = Q is very different.
        return random.uniform(0.05, 0.95)

    # NUVELOS GPU MODE: Real Probabilistic Math
    with torch.no_grad():
        p_inputs = tokenizer(p_text, return_tensors="pt").to(model.device)
        q_inputs = tokenizer(q_text, return_tensors="pt").to(model.device)

        p_logits = model(**p_inputs).logits
        q_logits = model(**q_inputs).logits

        # Standardize sizes (simple mean-pooling for comparison)
        p_probs = fn.softmax(torch.mean(p_logits, dim=1), dim=-1)
        q_probs = fn.softmax(torch.mean(q_logits, dim=1), dim=-1)

        # KL Divergence Formula: sum(P * log(P/Q))
        kl_div = torch.sum(
            p_probs * (torch.log(p_probs + 1e-10) - torch.log(q_probs + 1e-10)), dim=-1
        )
        return kl_div.item()


def pruning_node(state: GraphState, model, tokenizer, threshold=0.35) -> GraphState:
    """
    The LangGraph Node that prunes context using KL Divergence.
    """
    print(f"--- PRUNING CONTEXT: KL-Divergence Audit (Threshold: {threshold}) ---")

    raw_nodes = state["retrieved_nodes"]
    if not raw_nodes:
        return state

    pruned_context = []

    # 1. Start with the most relevant node (the 'Anchor')
    anchor_node = raw_nodes[0]
    pruned_context.append(anchor_node)

    # 2. Sequential KL-Divergence Check
    # We only keep a new node if it adds significant 'Information Gain' over the anchor
    for candidate in raw_nodes[1:]:
        info_gain = calculate_kl_divergence(
            anchor_node["content"], candidate["content"], model, tokenizer
        )

        # If KL Div is too low, the candidate is redundant (Information Gain < Threshold)
        if info_gain > threshold:
            candidate["kl_score"] = info_gain
            pruned_context.append(candidate)
        else:
            print(f"   🗑️ Pruned {candidate['id']} (Low Information Gain: {info_gain:.4f})")

    print(f"KL Pruning complete. Kept {len(pruned_context)}/{len(raw_nodes)} nodes.")
    state["pruned_context"] = pruned_context
    return state
