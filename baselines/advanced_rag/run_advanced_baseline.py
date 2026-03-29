"""Run Advanced RAG (CORTEX) queries via the web-ui API and capture eval fields."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import requests


def _safe_json(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_retrieved_ids(result: dict[str, Any]) -> list[str]:
    ids: list[str] = []

    for step in result.get("reasoning_steps", []) or []:
        for node_id in step.get("retrieved_nodes", []) or []:
            node = str(node_id).strip()
            if node:
                ids.append(node)

    for node in (result.get("graph_data", {}) or {}).get("nodes", []) or []:
        node_id = str(node.get("id", "")).strip()
        if node_id:
            ids.append(node_id)

    for citation in result.get("citations", []) or []:
        text = str(citation)
        if " - " in text:
            candidate = text.rsplit(" - ", 1)[-1].strip()
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

    snippets: list[str] = []
    graph_nodes = (result.get("graph_data", {}) or {}).get("nodes", []) or []
    for node in graph_nodes:
        preview = str(node.get("text_preview", "")).strip()
        if preview:
            snippets.append(preview)

    for citation in result.get("citations", []) or []:
        snippets.append(str(citation).strip())

    joined = "\n\n".join(snippet for snippet in snippets if snippet)
    return joined.strip()


def run_advanced_query(
    question: str,
    *,
    api_base_url: str,
    regulation: str,
    max_hops: int,
    enable_pruning: bool,
    enable_self_correction: bool,
    pruning_threshold: float,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Submit one query to advanced RAG and return eval-ready fields."""
    payload = {
        "question": question,
        "regulation": regulation,
        "max_hops": max_hops,
        "enable_pruning": enable_pruning,
        "enable_self_correction": enable_self_correction,
        "pruning_threshold": pruning_threshold,
    }

    started = time.time()

    submit = requests.post(f"{api_base_url}/query", json=payload, timeout=30)
    submit.raise_for_status()
    submit_data = _safe_json(submit)
    query_id = str(submit_data.get("query_id", "")).strip()
    if not query_id:
        raise RuntimeError("Advanced baseline did not return query_id")

    deadline = started + timeout_seconds
    last_payload: dict[str, Any] = {}

    while time.time() < deadline:
        status_resp = requests.get(f"{api_base_url}/query/{query_id}", timeout=30)
        status_resp.raise_for_status()
        last_payload = _safe_json(status_resp)

        status = str(last_payload.get("status", "")).lower().strip()
        if status in {"completed", "failed"}:
            break
        time.sleep(poll_interval_seconds)

    elapsed = time.time() - started

    status = str(last_payload.get("status", "unknown"))
    final_answer = str(last_payload.get("final_answer") or "").strip()
    retrieved_ids = _extract_retrieved_ids(last_payload)
    retrieved_context = _extract_retrieved_context(last_payload)

    return {
        "query_id": query_id,
        "status": status,
        "response": final_answer,
        "retrieved_ids": retrieved_ids,
        "retrieved_context": retrieved_context,
        "latency_seconds": round(elapsed, 3),
        "raw_result": last_payload,
        "error": str(last_payload.get("error") or "").strip() if status == "failed" else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one advanced RAG query")
    parser.add_argument("--question", required=True, help="Question text")
    parser.add_argument("--api-base-url", default="http://localhost:8000/api")
    parser.add_argument("--regulation", default="both", choices=["eu_ai_act", "dsa", "both"])
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--disable-pruning", action="store_true")
    parser.add_argument("--disable-self-correction", action="store_true")
    parser.add_argument("--pruning-threshold", type=float, default=0.45)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    result = run_advanced_query(
        args.question,
        api_base_url=args.api_base_url.rstrip("/"),
        regulation=args.regulation,
        max_hops=args.max_hops,
        enable_pruning=not args.disable_pruning,
        enable_self_correction=not args.disable_self_correction,
        pruning_threshold=args.pruning_threshold,
        poll_interval_seconds=args.poll_interval,
        timeout_seconds=args.timeout,
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
