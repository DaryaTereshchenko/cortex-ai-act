"""Legal-aware chunker for EU AI Act / DSA texts extracted from the Knowledge Graph.

Chunking strategy:
  - Respects semantic boundaries: Article, Recital, Annex section
  - Target chunk size: 500–1,000 tokens (configurable)
  - Overlap: 100–150 tokens (configurable)
  - Never splits mid-sentence
"""

from __future__ import annotations

import logging
import re
from typing import Any

from neo4j import GraphDatabase

from .config import Chunk, JudgeRAGConfig

log = logging.getLogger(__name__)

# ── Sentence boundary regex ─────────────────────────────────────────────────
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _tokenize_approx(text: str) -> list[str]:
    """Whitespace tokeniser (good enough for token-count estimates)."""
    return text.split()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences without breaking mid-sentence."""
    parts = _SENT_SPLIT.split(text)
    return [s.strip() for s in parts if s.strip()]


def _chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split *text* into chunks of roughly *chunk_size* tokens with *overlap*,
    aligned on sentence boundaries."""
    sentences = _split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(_tokenize_approx(sent))

        if current_len + sent_len > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Compute overlap: walk backwards until we have ~overlap tokens
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                s_len = len(_tokenize_approx(s))
                if overlap_len + s_len > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += s_len

            current_sentences = overlap_sents
            current_len = overlap_len

        current_sentences.append(sent)
        current_len += sent_len

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# ── KG text extraction ──────────────────────────────────────────────────────


def _extract_nodes(driver, database: str, regulation: str) -> list[dict[str, Any]]:
    """Pull every text-bearing node from Neo4j for the given regulation(s)."""
    reg_clause = ""
    params: dict[str, Any] = {}
    if regulation != "both":
        reg_clause = "WHERE n.regulation = $reg"
        params["reg"] = regulation

    queries = [
        # Articles (full_text)
        f"""
        MATCH (n:Article) {reg_clause}
        RETURN n.id AS id, 'Article' AS type, n.regulation AS regulation,
               n.full_text AS text, n.title AS title, n.number AS number,
               n.summary AS summary, n.key_topics AS key_topics
        """,
        # Paragraphs
        f"""
        MATCH (n:Paragraph) {reg_clause}
        RETURN n.id AS id, 'Paragraph' AS type, n.regulation AS regulation,
               n.text AS text, '' AS title, n.number AS number,
               '' AS summary, [] AS key_topics
        """,
        # SubParagraphs
        f"""
        MATCH (n:SubParagraph) {reg_clause}
        RETURN n.id AS id, 'SubParagraph' AS type, n.regulation AS regulation,
               n.text AS text, '' AS title, n.number AS number,
               '' AS summary, [] AS key_topics
        """,
        # Recitals
        f"""
        MATCH (n:Recital) {reg_clause}
        RETURN n.id AS id, 'Recital' AS type, n.regulation AS regulation,
               n.text AS text, '' AS title, n.number AS number,
               '' AS summary, [] AS key_topics
        """,
        # Definitions
        f"""
        MATCH (n:Definition) {reg_clause}
        RETURN n.id AS id, 'Definition' AS type, n.regulation AS regulation,
               n.definition_text AS text, n.term AS title, '' AS number,
               '' AS summary, [] AS key_topics
        """,
        # Annex sections
        f"""
        MATCH (n:AnnexSection) {reg_clause}
        RETURN n.id AS id, 'AnnexSection' AS type, n.regulation AS regulation,
               n.content AS text, n.title AS title, n.number AS number,
               '' AS summary, [] AS key_topics
        """,
    ]

    nodes: list[dict[str, Any]] = []
    with driver.session(database=database) as session:
        for q in queries:
            result = session.run(q, params)
            for record in result:
                data = record.data()
                if data.get("text"):
                    nodes.append(data)

    log.info("Extracted %d text nodes from KG (regulation=%s)", len(nodes), regulation)
    return nodes


def build_chunks(cfg: JudgeRAGConfig) -> list[Chunk]:
    """Connect to Neo4j, extract text nodes, and produce legal-aware chunks."""
    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password))
    try:
        nodes = _extract_nodes(driver, cfg.neo4j_database, cfg.regulation)
    finally:
        driver.close()

    all_chunks: list[Chunk] = []
    chunk_counter = 0

    for node in nodes:
        text = node["text"]
        source_type = node["type"]
        source_id = node["id"]
        regulation = node["regulation"] or ""

        # For short nodes (single paragraph / definition), keep as one chunk
        token_len = len(_tokenize_approx(text))
        if token_len <= cfg.chunk_size_tokens:
            chunk_counter += 1
            all_chunks.append(
                Chunk(
                    chunk_id=f"chunk_{chunk_counter}",
                    source_id=source_id,
                    source_type=source_type,
                    regulation=regulation,
                    text=text,
                    metadata={
                        "title": node.get("title") or "",
                        "number": node.get("number") or "",
                        "summary": node.get("summary") or "",
                        "key_topics": node.get("key_topics") or [],
                        "token_count": token_len,
                    },
                )
            )
        else:
            # Chunk larger text blocks respecting sentence boundaries
            sub_texts = _chunk_text(text, cfg.chunk_size_tokens, cfg.chunk_overlap_tokens)
            for i, sub in enumerate(sub_texts):
                chunk_counter += 1
                all_chunks.append(
                    Chunk(
                        chunk_id=f"chunk_{chunk_counter}",
                        source_id=source_id,
                        source_type=source_type,
                        regulation=regulation,
                        text=sub,
                        metadata={
                            "title": node.get("title") or "",
                            "number": node.get("number") or "",
                            "summary": node.get("summary") or "",
                            "key_topics": node.get("key_topics") or [],
                            "sub_chunk_index": i,
                            "total_sub_chunks": len(sub_texts),
                            "token_count": len(_tokenize_approx(sub)),
                        },
                    )
                )

    log.info("Built %d chunks from %d source nodes", len(all_chunks), len(nodes))
    return all_chunks
