"""Run Advanced RAG (CORTEX) queries by calling run_cortex_engine directly.

Bypasses the web-ui gateway and the reasoning-engine HTTP API — calls the
orchestrator in-process so that only Neo4j (via the KG service at :8001) is
needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add the reasoning-engine source to the import path so we can reuse
# run_cortex_engine and its helpers without duplicating code.
_RE_DIR = str(Path(__file__).resolve().parents[2] / "services" / "reasoning-engine")
if _RE_DIR not in sys.path:
    sys.path.insert(0, _RE_DIR)

from main_orchestrator import run_cortex_engine  # noqa: E402


def _extract_retrieved_ids(result: dict[str, Any]) -> list[str]:
    ids: list[str] = []

    for step in result.get("reasoning_steps", []) or []:
        for node_id in step.get("retrieved_nodes", []) or []:
            node = str(node_id).strip()
            if node:
                ids.append(node)

    for citation in result.get("citations", []) or []:
        text = str(citation)
        # Citations look like "eu_ai_act_art_5 (EU_AI_ACT)"
        candidate = text.split("(")[0].strip() if "(" in text else text.strip()
        if candidate:
            ids.append(candidate)

    deduped: list[str] = []
    seen: set[str] = set()
    for node_id in ids:
        if node_id not in seen:
            seen.add(node_id)
            deduped.append(node_id)
    return deduped


def _extract_retrieved_context(result: dict[str, Any]) -> str:
    candidates = [
        result.get("retrieved_context"),
        (result.get("metrics", {}) or {}).get("retrieved_context"),
        result.get("context"),
    ]

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    # Fallback: build context from citations
    snippets: list[str] = []
    for citation in result.get("citations", []) or []:
        snippets.append(str(citation).strip())

    joined = "\n\n".join(snippet for snippet in snippets if snippet)
    return joined.strip()


def run_advanced_query(
    question: str,
    *,
    regulation: str = "both",
    max_hops: int = 3,
    enable_pruning: bool = False,
    enable_self_correction: bool = False,
    pruning_threshold: float = 0.45,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Call run_cortex_engine directly (no HTTP, no Docker gateway needed)."""
    started = time.time()

    try:
        result = run_cortex_engine(
            question,
            max_hops=max_hops,
            enable_pruning=enable_pruning,
            enable_self_correction=enable_self_correction,
            pruning_threshold=pruning_threshold,
        )
    except Exception as exc:
        return {
            "query_id": "",
            "status": "failed",
            "response": "",
            "retrieved_ids": [],
            "retrieved_context": "",
            "latency_seconds": round(time.time() - started, 3),
            "raw_result": {},
            "error": str(exc),
        }

    elapsed = time.time() - started

    status = str(result.get("status", "completed"))
    final_answer = str(result.get("final_answer") or "").strip()
    retrieved_ids = _extract_retrieved_ids(result)
    retrieved_context = _extract_retrieved_context(result)

    return {
        "query_id": str(result.get("query_id", "")),
        "status": status,
        "response": final_answer,
        "retrieved_ids": retrieved_ids,
        "retrieved_context": retrieved_context,
        "latency_seconds": round(elapsed, 3),
        "raw_result": result,
        "error": str(result.get("error") or "").strip() if status == "failed" else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one advanced RAG query")
    parser.add_argument("--question", required=True, help="Question text")
    parser.add_argument("--regulation", default="both", choices=["eu_ai_act", "dsa", "both"])
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--disable-pruning", action="store_true")
    parser.add_argument("--disable-self-correction", action="store_true")
    parser.add_argument("--pruning-threshold", type=float, default=0.45)
    args = parser.parse_args()

    result = run_advanced_query(
        args.question,
        regulation=args.regulation,
        max_hops=args.max_hops,
        enable_pruning=not args.disable_pruning,
        enable_self_correction=not args.disable_self_correction,
        pruning_threshold=args.pruning_threshold,
    )

    printable = {
        "status": result["status"],
        "response_preview": result["response"][:300],
        "retrieved_ids": result["retrieved_ids"],
        "retrieved_context_chars": len(result["retrieved_context"]),
        "latency_seconds": result["latency_seconds"],
        "error": result["error"],
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
