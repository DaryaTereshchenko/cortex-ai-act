#!/usr/bin/env python3
"""CLI entry point for the Judge RAG pipeline.

Usage examples
--------------
# Basic — uses default llama3.1:8b for all models
python -m baselines.judge_rag.run_judge_rag \\
    --question "What does Article 5 of the AI Act prohibit?"

# Pick different Ollama models for each stage
python -m baselines.judge_rag.run_judge_rag \\
    --question "What are the transparency obligations for high-risk AI?" \\
    --retrieval-model mistral:7b \\
    --generation-model llama3.1:8b \\
    --judge-model llama3.1:70b

# Disable query rewriting, target DSA only
python -m baselines.judge_rag.run_judge_rag \\
    --question "What are the obligations for online platforms?" \\
    --no-query-rewrite \\
    --regulation dsa

# Custom chunking parameters
python -m baselines.judge_rag.run_judge_rag \\
    --question "Explain conformity assessment" \\
    --chunk-size 500 \\
    --chunk-overlap 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure repository root is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.judge_rag.config import JudgeRAGConfig
from baselines.judge_rag.pipeline import run_judge_rag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Judge RAG pipeline — hybrid retrieval with LLM-as-judge loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument(
        "--question", "-q",
        required=True,
        help="The legal question to answer",
    )

    # Ollama models — any model name available on the local Ollama server
    p.add_argument(
        "--retrieval-model",
        default="llama3.1:8b",
        help="Ollama model for query rewriting (default: llama3.1:8b)",
    )
    p.add_argument(
        "--generation-model",
        default="llama3.1:8b",
        help="Ollama model for answer generation (default: llama3.1:8b)",
    )
    p.add_argument(
        "--judge-model",
        default="llama3.1:8b",
        help="Ollama model for judge evaluation (default: llama3.1:8b)",
    )
    p.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )

    # Retrieval tuning
    p.add_argument(
        "--chunk-size", type=int, default=750,
        help="Target chunk size in tokens (default: 750, range: 500-1000)",
    )
    p.add_argument(
        "--chunk-overlap", type=int, default=125,
        help="Chunk overlap in tokens (default: 125, range: 100-150)",
    )
    p.add_argument(
        "--bm25-top-k", type=int, default=25,
        help="BM25 initial recall pool size (default: 25)",
    )
    p.add_argument(
        "--rerank-top-k", type=int, default=10,
        help="Semantic re-ranking pool size (default: 10)",
    )
    p.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for re-ranking (default: all-MiniLM-L6-v2)",
    )

    # Query rewriting
    p.add_argument(
        "--no-query-rewrite",
        action="store_true",
        help="Disable LLM query rewriting",
    )

    # Judge parameters
    p.add_argument(
        "--max-attempts", type=int, default=3,
        help="Maximum judge re-generation attempts (default: 3)",
    )
    p.add_argument(
        "--judge-threshold", type=float, default=7.0,
        help="Minimum judge score to accept answer (1-10, default: 7.0)",
    )

    # Regulation filter
    p.add_argument(
        "--regulation",
        choices=["eu_ai_act", "dsa", "both"],
        default="eu_ai_act",
        help="Which regulation to search (default: eu_ai_act)",
    )

    # Neo4j connection
    p.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-password", default="changeme")

    # Output
    p.add_argument(
        "--output", "-o",
        help="Write JSON result to this file (default: stdout)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    cfg = JudgeRAGConfig(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        retrieval_model=args.retrieval_model,
        generation_model=args.generation_model,
        judge_model=args.judge_model,
        ollama_base_url=args.ollama_base_url,
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_tokens=args.chunk_overlap,
        bm25_top_k=args.bm25_top_k,
        rerank_top_k=args.rerank_top_k,
        embedding_model=args.embedding_model,
        enable_query_rewriting=not args.no_query_rewrite,
        max_judge_attempts=args.max_attempts,
        judge_accept_threshold=args.judge_threshold,
        regulation=args.regulation,
    )

    result = run_judge_rag(args.question, cfg)

    output_json = json.dumps(result, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output_json)
        print(f"Result written to {args.output}")
    else:
        print("\n" + "=" * 80)
        print("JUDGE RAG RESULT")
        print("=" * 80)
        print(f"\nQuery:        {result['query']}")
        print(f"Uncertainty:  {result['uncertainty']}")
        print(f"Judge score:  {result['judge_score']:.1f}/10")
        print(f"Attempts:     {result['attempts_used']}/{cfg.max_judge_attempts}")
        print(f"Retrieval ↔:  {result['retrieval_similarity']:.3f}")
        print(f"Latency:      {result['latency_seconds']}s")
        print(f"Retrieved:    {len(result['retrieved_ids'])} chunks")

        if result["rewritten_queries"]:
            print(f"\nSub-queries:")
            for sq in result["rewritten_queries"]:
                print(f"  • {sq}")

        print(f"\nAnswer:\n{result['answer']}")

        if result["uncertainty"] != "low":
            print(f"\n⚠ Uncertainty: {result['uncertainty']} — "
                  f"judge feedback: {result['judge_feedback']}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
