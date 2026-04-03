from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "changeme"


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _normalize(vec: list[float]) -> list[float]:
    norm = _l2_norm(vec)
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _load_embedding_model():
    try:
        st_module = __import__("sentence_transformers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'sentence-transformers'. Install with: pip install -r baselines/requirements.txt"
        ) from exc
    return st_module.SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _load_article_embeddings() -> tuple[list[str], list[str], list[list[float]], Any]:
    query = """
    MATCH (c:DocumentChunk)
    WHERE c.chunk_id IS NOT NULL AND c.text IS NOT NULL
    RETURN c.chunk_id AS id, c.text AS text
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

    if not texts:
        return ids, texts, [], None

    model = _load_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False)
    embeddings: list[list[float]] = []
    for vec in vectors:
        embeddings.append(_normalize([float(v) for v in vec]))

    return ids, texts, embeddings, model


def run_dense_embedding_rag_benchmark(query_text: str, top_k: int = 5) -> dict[str, Any]:
    ids, texts, embeddings, model = _load_article_embeddings()
    if not embeddings:
        return {
            "nodes_found": 0,
            "retrieved_ids": [],
            "retrieved_context": "",
            "total_word_count": 0,
        }

    query_vec = model.encode([query_text], show_progress_bar=False)[0]
    query_embedding = _normalize([float(v) for v in query_vec])

    ranked = sorted(
        enumerate(embeddings),
        key=lambda item: _dot(query_embedding, item[1]),
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
    result = run_dense_embedding_rag_benchmark(sample_query, top_k=5)
    print("[DENSE EMBEDDING BASELINE]")
    print(f"Nodes Retrieved: {result['nodes_found']}")
    print(f"Retrieved IDs: {result['retrieved_ids']}")
    print(f"Total Context Size: {result['total_word_count']} words")
