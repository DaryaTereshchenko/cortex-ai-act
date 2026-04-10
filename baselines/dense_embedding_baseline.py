from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "changeme"

# Default and supported embedding models
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LEGAL_EMBEDDING_MODEL = "law-ai/InLegalBERT"


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _normalize(vec: list[float]) -> list[float]:
    norm = _l2_norm(vec)
    if norm == 0:
        return vec
    return [v / norm for v in vec]


_model_cache: dict[str, Any] = {}


def _load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        st_module = __import__("sentence_transformers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'sentence-transformers'. Install with: pip install -r baselines/requirements.txt"
        ) from exc
    model = st_module.SentenceTransformer(model_name)
    _model_cache[model_name] = model
    return model


# Cache per model name to avoid re-encoding articles for each query
_article_cache: dict[str, tuple[list[str], list[str], list[list[float]], Any]] = {}


def _load_article_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[list[str], list[str], list[list[float]], Any]:
    if model_name in _article_cache:
        return _article_cache[model_name]

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

    if not texts:
        result = (ids, texts, [], None)
        _article_cache[model_name] = result
        return result

    model = _load_embedding_model(model_name)
    vectors = model.encode(texts, show_progress_bar=False)
    embeddings: list[list[float]] = []
    for vec in vectors:
        embeddings.append(_normalize([float(v) for v in vec]))

    result = (ids, texts, embeddings, model)
    _article_cache[model_name] = result
    return result


def run_dense_embedding_rag_benchmark(
    query_text: str,
    top_k: int = 5,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    ids, texts, embeddings, model = _load_article_embeddings(model_name)
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
