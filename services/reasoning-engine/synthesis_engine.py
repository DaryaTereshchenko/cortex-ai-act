import logging

import ollama

from engine_schema import GraphState

log = logging.getLogger(__name__)

# --- INNOVATION 3: CONTEXT-AWARE SYNTHESIS ---

_SYSTEM_PROMPT = """\
You are a legal expert on the EU AI Act (Regulation (EU) 2024/1689) and the Digital Services Act \
(Regulation (EU) 2022/2065).

Answer the user's question using ONLY the legal context provided below.
Rules:
  1. Cite specific Article numbers, Recital numbers, or Annex sections where applicable.
  2. If the context does not contain enough information, say so — never fabricate.
  3. Use precise legal language. Keep the answer concise but thorough.
  4. Structure the answer with bullet points or numbered items when listing obligations.
"""


def synthesis_node(
    state: GraphState,
    *,
    synthesis_model: str = "",
    ollama_base_url: str = "http://localhost:11434",
) -> GraphState:
    """
    Innovation 3: Context-Aware Synthesis.

    When *synthesis_model* is provided, generates the answer via Ollama LLM.
    Otherwise falls back to the original template-based synthesis.
    """
    print("--- SYNTHESIZING FINAL ANSWER ---")

    if not state["pruned_context"]:
        state["final_answer"] = "No relevant legal context found to support this query."
        return state

    if synthesis_model:
        state = _llm_synthesis(state, synthesis_model, ollama_base_url)
    else:
        state = _template_synthesis(state)

    return state


def _llm_synthesis(
    state: GraphState,
    model: str,
    ollama_base_url: str,
) -> GraphState:
    """Generate the final answer using an Ollama LLM."""
    # Build context block from pruned nodes
    context_parts: list[str] = []
    for i, node in enumerate(state["pruned_context"], 1):
        header = f"[{i}] {node['node_type']} {node['id']}"
        context_parts.append(f"{header}\n{node['content']}")
    context_block = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"## Retrieved Legal Context\n\n{context_block}\n\n"
        f"## Question\n\n{state['query']}"
    )

    client = ollama.Client(host=ollama_base_url)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={"temperature": 0.2, "num_predict": 2048},
    )

    state["final_answer"] = response["message"]["content"].strip()
    state["reasoning_trace"].append(
        f"Synthesizer: LLM response generated via {model}."
    )
    log.info("LLM synthesis (%s): %d chars", model, len(state["final_answer"]))
    return state


def _template_synthesis(state: GraphState) -> GraphState:
    """Original template-based synthesis (no LLM)."""
    top_nodes = state["pruned_context"][:3]
    intro = f"Analysis based on optimized {state['retrieved_nodes'][0]['regulation'].upper()} context:\n"

    body_parts = []
    for node in top_nodes:
        snippet = node["content"][:1000].strip() + "..."
        body_parts.append(f"• SOURCE {node['id'].upper()}: {snippet}")

    optimization_val = state["metrics"].get("entropy_reduction", 0.0) * 100
    conclusion = f"\n\nContext optimized for high-fidelity response. (Efficiency Gain: {optimization_val:.1f}%)"

    state["final_answer"] = intro + "\n".join(body_parts) + conclusion
    state["reasoning_trace"].append(
        "Synthesizer: Response generated from semantically optimized context."
    )
    return state
