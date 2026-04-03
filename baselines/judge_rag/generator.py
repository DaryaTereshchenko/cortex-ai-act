"""Answer generation — feeds retrieved context + original query into an Ollama LLM."""

from __future__ import annotations

import logging

import ollama

from .config import JudgeRAGConfig, RetrievedChunk

log = logging.getLogger(__name__)

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


def _build_context_block(retrieved: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts: list[str] = []
    for i, rc in enumerate(retrieved, 1):
        c = rc["chunk"]
        header = f"[{i}] {c['source_type']} {c['source_id']}"
        if c["metadata"].get("title"):
            header += f" — {c['metadata']['title']}"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    retrieved: list[RetrievedChunk],
    cfg: JudgeRAGConfig,
    *,
    prior_feedback: str | None = None,
) -> str:
    """Generate an answer using the configured Ollama generation model.

    If *prior_feedback* is provided (from the judge), it is appended to the
    user message so the LLM can self-correct.
    """
    context_block = _build_context_block(retrieved)

    user_message = (
        f"## Retrieved Legal Context\n\n{context_block}\n\n"
        f"## Question\n\n{query}"
    )

    if prior_feedback:
        user_message += (
            f"\n\n## Previous Attempt Feedback\n\n"
            f"A reviewer flagged the following issues with your prior answer. "
            f"Please address them:\n{prior_feedback}"
        )

    response = ollama.chat(
        model=cfg.generation_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={"temperature": 0.2, "num_predict": 2048},
    )

    answer = response["message"]["content"].strip()
    log.info(
        "Generated answer (%d chars) using model '%s'",
        len(answer),
        cfg.generation_model,
    )
    return answer
