"""LangGraph orchestrator — wires the Judge RAG pipeline stages into a state-machine graph.

Nodes:
  1. rewrite_query    — LLM query rewriting (optional)
  2. retrieve         — BM25 → semantic re-rank → adaptive-K
  3. score_retrieval  — cosine similarity eval metric
  4. generate         — LLM answer generation
  5. judge            — LLM-as-judge evaluation
  6. decide           — conditional edge: accept or retry (max 3 attempts)

Edges:
  rewrite_query → retrieve → score_retrieval → generate → judge → decide
  decide →(retry)→ generate
  decide →(accept)→ END
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import END, StateGraph

from .chunker import build_chunks
from .config import JudgeRAGConfig, PipelineState
from .generator import generate_answer
from .judge import evaluate_answer, should_retry, uncertainty_label
from .query_rewriter import rewrite_query
from .retriever import HybridRetriever

log = logging.getLogger(__name__)


# ── Node functions ──────────────────────────────────────────────────────────


def _rewrite_node(state: PipelineState) -> dict[str, Any]:
    """Rewrite the user query into focused sub-queries."""
    cfg: JudgeRAGConfig = state["metrics"]["_cfg"]
    sub_queries = rewrite_query(state["query"], cfg)
    return {"rewritten_queries": sub_queries}


def _retrieve_node(state: PipelineState) -> dict[str, Any]:
    """Run hybrid retrieval over the chunk index."""
    retriever: HybridRetriever = state["metrics"]["_retriever"]
    queries = state["rewritten_queries"] or [state["query"]]
    retrieved = retriever.retrieve(queries)
    return {"retrieved": retrieved}


def _score_retrieval_node(state: PipelineState) -> dict[str, Any]:
    """Score retrieved chunks against the original query (eval metric)."""
    retriever: HybridRetriever = state["metrics"]["_retriever"]
    sim = retriever.score_against_query(state["query"], state["retrieved"])
    return {"retrieval_similarity": sim}


def _generate_node(state: PipelineState) -> dict[str, Any]:
    """Generate an answer using the configured LLM."""
    cfg: JudgeRAGConfig = state["metrics"]["_cfg"]
    prior_feedback = state.get("judge_feedback") if state["attempt"] > 1 else None
    answer = generate_answer(
        state["query"], state["retrieved"], cfg, prior_feedback=prior_feedback,
    )
    return {"generated_answer": answer}


def _judge_node(state: PipelineState) -> dict[str, Any]:
    """Run the LLM judge on the generated answer."""
    cfg: JudgeRAGConfig = state["metrics"]["_cfg"]
    result = evaluate_answer(
        state["query"], state["retrieved"], state["generated_answer"], cfg,
    )
    accepted = not should_retry(result, cfg)
    unc = uncertainty_label(result)

    log.info(
        "Judge attempt %d/%d — score %.1f (%s) — %s",
        state["attempt"],
        state["max_attempts"],
        result.overall,
        unc,
        "ACCEPTED" if accepted else "RETRY",
    )

    return {
        "judge_score": result.overall,
        "judge_feedback": result.feedback,
        "is_accepted": accepted,
        "uncertainty": unc,
        "attempt": state["attempt"] + 1,
        "metrics": {
            **state["metrics"],
            f"judge_attempt_{state['attempt']}": {
                "relevance": result.relevance,
                "faithfulness": result.faithfulness,
                "completeness": result.completeness,
                "precision": result.precision,
                "overall": result.overall,
                "feedback": result.feedback,
            },
        },
    }


def _decide_edge(state: PipelineState) -> str:
    """Conditional edge: accept the answer or loop back for regeneration."""
    if state["is_accepted"]:
        return "accept"
    if state["attempt"] >= state["max_attempts"]:
        log.warning(
            "Max attempts (%d) reached — returning best answer with uncertainty '%s'",
            state["max_attempts"],
            state["uncertainty"],
        )
        return "accept"
    return "retry"


# ── Graph construction ──────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Build the LangGraph state-machine for the Judge RAG pipeline."""
    graph = StateGraph(PipelineState)

    graph.add_node("rewrite_query", _rewrite_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("score_retrieval", _score_retrieval_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("judge", _judge_node)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "score_retrieval")
    graph.add_edge("score_retrieval", "generate")
    graph.add_edge("generate", "judge")

    graph.add_conditional_edges(
        "judge",
        _decide_edge,
        {"accept": END, "retry": "generate"},
    )

    return graph


# ── Public runner ───────────────────────────────────────────────────────────


def run_judge_rag(query: str, cfg: JudgeRAGConfig | None = None) -> dict[str, Any]:
    """End-to-end Judge RAG pipeline execution.

    Returns a result dict with:
      - answer, uncertainty, judge_score, judge_feedback
      - retrieval_similarity (eval metric)
      - retrieved chunk ids
      - per-attempt judge scores
      - latency
    """
    cfg = cfg or JudgeRAGConfig()
    start = time.time()

    # 1. Build chunks from KG
    log.info("=== Judge RAG Pipeline Start ===")
    log.info("Models — retrieval: %s | generation: %s | judge: %s",
             cfg.retrieval_model, cfg.generation_model, cfg.judge_model)

    chunks = build_chunks(cfg)
    retriever = HybridRetriever(cfg, chunks)

    # 2. Compile and run the graph
    graph = build_graph()
    app = graph.compile()

    initial_state: PipelineState = {
        "query": query,
        "rewritten_queries": [],
        "chunks": chunks,
        "retrieved": [],
        "retrieval_similarity": 0.0,
        "generated_answer": "",
        "judge_score": 0.0,
        "judge_feedback": "",
        "attempt": 1,
        "max_attempts": cfg.max_judge_attempts,
        "is_accepted": False,
        "uncertainty": "high",
        "metrics": {
            "_cfg": cfg,
            "_retriever": retriever,
        },
    }

    final_state = app.invoke(initial_state)

    elapsed = round(time.time() - start, 2)

    # 3. Build clean output
    retrieved_ids = [rc["chunk"]["source_id"] for rc in final_state["retrieved"]]
    retrieved_context = "\n\n".join(rc["chunk"]["text"] for rc in final_state["retrieved"])

    # Remove internal objects from metrics before returning
    clean_metrics = {
        k: v for k, v in final_state["metrics"].items() if not k.startswith("_")
    }

    result = {
        "query": query,
        "answer": final_state["generated_answer"],
        "uncertainty": final_state["uncertainty"],
        "judge_score": final_state["judge_score"],
        "judge_feedback": final_state["judge_feedback"],
        "attempts_used": final_state["attempt"] - 1,
        "retrieval_similarity": final_state["retrieval_similarity"],
        "retrieved_ids": list(dict.fromkeys(retrieved_ids)),  # dedupe, preserve order
        "retrieved_context": retrieved_context,
        "rewritten_queries": final_state["rewritten_queries"],
        "latency_seconds": elapsed,
        "judge_details": clean_metrics,
        "config": {
            "retrieval_model": cfg.retrieval_model,
            "generation_model": cfg.generation_model,
            "judge_model": cfg.judge_model,
            "chunk_size": cfg.chunk_size_tokens,
            "chunk_overlap": cfg.chunk_overlap_tokens,
            "regulation": cfg.regulation,
        },
    }

    log.info(
        "=== Pipeline Complete — score=%.1f uncertainty=%s attempts=%d latency=%.1fs ===",
        result["judge_score"],
        result["uncertainty"],
        result["attempts_used"],
        elapsed,
    )
    return result
