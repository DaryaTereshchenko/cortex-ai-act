"""Hybrid retriever: BM25 → Semantic Re-ranking → Adaptive-K selection.

Pipeline:
  1. BM25 keyword recall over all chunks
  2. BERT / Legal-BERT semantic re-ranking (sentence-transformers)
  3. Adaptive-K — dynamic number of chunks based on query complexity and score density
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

from baselines.model_registry import get_model as _registry_get_model
from .config import Chunk, JudgeRAGConfig, RetrievedChunk

log = logging.getLogger(__name__)


class HybridRetriever:
    """BM25 + dense re-ranker with adaptive-K chunk selection."""

    def __init__(self, cfg: JudgeRAGConfig, chunks: list[Chunk]) -> None:
        self.cfg = cfg
        self.chunks = chunks

        # Tokenised corpus for BM25
        log.info("Building BM25 index over %d chunks …", len(chunks))
        self._corpus_tokens = [c["text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

        # Dense encoder — shared via centralized registry
        log.info("Loading embedding model '%s' …", cfg.embedding_model)
        self._encoder = _registry_get_model(cfg.embedding_model)

        # Pre-compute chunk embeddings
        log.info("Encoding chunk embeddings …")
        texts = [c["text"] for c in chunks]
        self._chunk_embeddings = self._encoder.encode(
            texts, convert_to_tensor=True, show_progress_bar=True,
        )

    # ── public API ──────────────────────────────────────────────────────────

    def retrieve(self, queries: list[str]) -> list[RetrievedChunk]:
        """Run the full hybrid pipeline for one or more (rewritten) queries.

        Returns de-duplicated chunks sorted by combined score.
        """
        # 1. BM25 recall — merge scores across all sub-queries
        bm25_scores = np.zeros(len(self.chunks))
        for q in queries:
            tokens = q.lower().split()
            scores = self._bm25.get_scores(tokens)
            bm25_scores = np.maximum(bm25_scores, scores)

        # Take top-K by BM25
        bm25_top_indices = np.argsort(bm25_scores)[::-1][: self.cfg.bm25_top_k]

        # 2. Semantic re-ranking of the BM25 shortlist
        query_embedding = self._encoder.encode(
            " ".join(queries), convert_to_tensor=True,
        )

        candidates: list[RetrievedChunk] = []
        for idx in bm25_top_indices:
            idx = int(idx)
            chunk = self.chunks[idx]
            bm25_s = float(bm25_scores[idx])
            sem_s = float(util.cos_sim(query_embedding, self._chunk_embeddings[idx]))
            combined = 0.4 * self._norm_bm25(bm25_s, bm25_scores) + 0.6 * sem_s
            candidates.append(
                RetrievedChunk(
                    chunk=chunk,
                    bm25_score=bm25_s,
                    semantic_score=sem_s,
                    combined_score=combined,
                )
            )

        candidates.sort(key=lambda c: c["combined_score"], reverse=True)

        # 3. Adaptive-K selection
        k = self._adaptive_k(queries, candidates)
        selected = candidates[:k]

        log.info(
            "Hybrid retrieval: BM25 top-%d → rerank top-%d → adaptive K=%d",
            self.cfg.bm25_top_k,
            len(candidates),
            k,
        )
        return selected

    def score_against_query(
        self, query: str, retrieved: list[RetrievedChunk],
    ) -> float:
        """Compute average cosine similarity of retrieved chunks against the original query."""
        if not retrieved:
            return 0.0
        query_emb = self._encoder.encode(query, convert_to_tensor=True)
        scores = []
        for rc in retrieved:
            chunk_emb = self._encoder.encode(rc["chunk"]["text"], convert_to_tensor=True)
            scores.append(float(util.cos_sim(query_emb, chunk_emb)))
        return float(np.mean(scores))

    # ── internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _norm_bm25(score: float, all_scores: np.ndarray) -> float:
        """Min-max normalise a BM25 score to [0, 1]."""
        mn, mx = float(all_scores.min()), float(all_scores.max())
        if mx - mn < 1e-9:
            return 0.5
        return (score - mn) / (mx - mn)

    def _adaptive_k(
        self, queries: list[str], candidates: list[RetrievedChunk],
    ) -> int:
        """Decide how many chunks to keep based on query complexity and score density."""
        # Query complexity heuristic: number of distinct legal terms / sub-queries
        total_tokens = sum(len(q.split()) for q in queries)
        complexity = min(1.0, total_tokens / 30.0)  # normalise

        # Score density: how quickly do scores drop off?
        if len(candidates) < 2:
            density = 1.0
        else:
            top_score = candidates[0]["combined_score"]
            drop_idx = next(
                (
                    i
                    for i, c in enumerate(candidates)
                    if c["combined_score"] < 0.5 * top_score
                ),
                len(candidates),
            )
            density = drop_idx / len(candidates)

        # Blend into a K value
        raw_k = self.cfg.adaptive_k_min + (self.cfg.adaptive_k_max - self.cfg.adaptive_k_min) * (
            0.5 * complexity + 0.5 * density
        )
        k = max(self.cfg.adaptive_k_min, min(self.cfg.adaptive_k_max, int(math.ceil(raw_k))))

        # Never return more than we have
        return min(k, len(candidates))
