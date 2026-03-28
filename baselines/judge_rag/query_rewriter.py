"""LLM-based query rewriting — turns an ambiguous user query into
one or more specific legal sub-queries using an Ollama model.
"""

from __future__ import annotations

import json
import logging
import re

import ollama

from .config import JudgeRAGConfig

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a legal research assistant specialising in the EU AI Act and Digital Services Act.
Given a user query, rewrite it into 1-3 precise legal sub-queries that would retrieve the most \
relevant regulation text. Each sub-query should:
  • Use specific legal terminology (e.g. "high-risk AI system", "conformity assessment")
  • Target a distinct aspect of the original question
  • Include synonyms for key legal terms where relevant

Return ONLY a JSON array of strings, e.g.:
["sub-query 1", "sub-query 2"]

Do not include any other text or explanation.\
"""


def rewrite_query(query: str, cfg: JudgeRAGConfig) -> list[str]:
    """Use the retrieval LLM to expand *query* into focused sub-queries.

    Falls back to the original query on any failure.
    """
    if not cfg.enable_query_rewriting:
        return [query]

    try:
        response = ollama.chat(
            model=cfg.retrieval_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.3, "num_predict": 512},
        )
        content = response["message"]["content"].strip()

        # Parse JSON array from response (tolerant of markdown fences)
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*$", "", content)
        parsed = json.loads(content)

        if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
            sub_queries = [s.strip() for s in parsed if s.strip()]
            if sub_queries:
                log.info("Query rewritten into %d sub-queries", len(sub_queries))
                return sub_queries

    except Exception as exc:
        log.warning("Query rewriting failed (%s), using original query", exc)

    return [query]
