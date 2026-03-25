from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "changeme"


TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _build_bm25(tokenized: list[list[str]]):
    try:
        rank_bm25 = __import__("rank_bm25")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'rank-bm25'. Install with: pip install -r baselines/requirements.txt"
        ) from exc
    return rank_bm25.BM25Okapi(tokenized)


@lru_cache(maxsize=1)
def _load_article_corpus() -> tuple[list[str], list[str], list[list[str]], Any]:
    query = """
    MATCH (a:Article)
    WHERE a.id IS NOT NULL AND a.full_text IS NOT NULL
    RETURN a.id AS id, a.full_text AS text
    """

    ids: list[str] = []
    texts: list[str] = []

    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver, driver.session() as session:
        rows = session.run(query)
        for row in rows:
            node_id = str(row.get("id") or "").strip()
            text = str(row.get("text") or "").strip()
            if not node_id or not text:
                continue
            ids.append(node_id)
            texts.append(text)

    tokenized = [_tokenize(t) for t in texts]
    bm25 = _build_bm25(tokenized)
    return ids, texts, tokenized, bm25


def run_bm25_rag_benchmark(query_text: str, top_k: int = 5) -> dict[str, Any]:
    ids, texts, _tokenized, bm25 = _load_article_corpus()
    if not texts:
        return {
            "nodes_found": 0,
            "retrieved_ids": [],
            "retrieved_context": "",
            "total_word_count": 0,
        }

    query_tokens = _tokenize(query_text)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        enumerate(scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )[: max(1, int(top_k))]

    retrieved_ids = [ids[i] for i, _ in ranked]
    retrieved_texts = [texts[i] for i, _ in ranked]
    context_text = "\n\n".join(retrieved_texts)

    return {
        "nodes_found": len(retrieved_texts),
        "retrieved_ids": retrieved_ids,
        "retrieved_context": context_text,
        "total_word_count": len(context_text.split()),
    }


if __name__ == "__main__":
    sample_query = "transparency obligations AI systems"
    result = run_bm25_rag_benchmark(sample_query, top_k=5)
    print("[BM25 BASELINE]")
    print(f"Nodes Retrieved: {result['nodes_found']}")
    print(f"Retrieved IDs: {result['retrieved_ids']}")
    print(f"Total Context Size: {result['total_word_count']} words")
