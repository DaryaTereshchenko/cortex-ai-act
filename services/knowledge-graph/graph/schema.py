"""Neo4j schema: constraints, indexes, and Cypher setup for the knowledge graph."""

from __future__ import annotations

import logging

from graph.connection import Neo4jConnection

log = logging.getLogger(__name__)

# ── Uniqueness constraints ──────────────────────────────────────────────────
CONSTRAINTS = [
    "CREATE CONSTRAINT reg_id IF NOT EXISTS FOR (n:Regulation) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT rec_id IF NOT EXISTS FOR (n:Recital) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT ch_id IF NOT EXISTS FOR (n:Chapter) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT sec_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT art_id IF NOT EXISTS FOR (n:Article) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT par_id IF NOT EXISTS FOR (n:Paragraph) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT anx_id IF NOT EXISTS FOR (n:Annex) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT anx_sec_id IF NOT EXISTS FOR (n:AnnexSection) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT def_id IF NOT EXISTS FOR (n:Definition) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT subpar_id IF NOT EXISTS FOR (n:SubParagraph) REQUIRE n.id IS UNIQUE",
]

# ── Full-text and lookup indexes ────────────────────────────────────────────
INDEXES = [
    # B-tree indexes for fast lookups
    "CREATE INDEX art_reg IF NOT EXISTS FOR (n:Article) ON (n.regulation)",
    "CREATE INDEX par_reg IF NOT EXISTS FOR (n:Paragraph) ON (n.regulation)",
    "CREATE INDEX rec_reg IF NOT EXISTS FOR (n:Recital) ON (n.regulation)",
    "CREATE INDEX ch_reg IF NOT EXISTS FOR (n:Chapter) ON (n.regulation)",
    "CREATE INDEX def_reg IF NOT EXISTS FOR (n:Definition) ON (n.regulation)",
    "CREATE INDEX subpar_reg IF NOT EXISTS FOR (n:SubParagraph) ON (n.regulation)",
    "CREATE INDEX art_number IF NOT EXISTS FOR (n:Article) ON (n.number)",
    "CREATE INDEX rec_number IF NOT EXISTS FOR (n:Recital) ON (n.number)",
    # Full-text search indexes for RAG queries
    (
        "CREATE FULLTEXT INDEX article_fulltext IF NOT EXISTS "
        "FOR (n:Article) ON EACH [n.title, n.summary, n.full_text]"
    ),
    ("CREATE FULLTEXT INDEX paragraph_fulltext IF NOT EXISTS FOR (n:Paragraph) ON EACH [n.text]"),
    ("CREATE FULLTEXT INDEX recital_fulltext IF NOT EXISTS FOR (n:Recital) ON EACH [n.text]"),
    (
        "CREATE FULLTEXT INDEX definition_fulltext IF NOT EXISTS "
        "FOR (n:Definition) ON EACH [n.term, n.definition_text]"
    ),
    (
        "CREATE FULLTEXT INDEX subparagraph_fulltext IF NOT EXISTS "
        "FOR (n:SubParagraph) ON EACH [n.text]"
    ),
]


def apply_schema(conn: Neo4jConnection) -> None:
    """Create all constraints and indexes (idempotent)."""
    for stmt in CONSTRAINTS:
        try:
            conn.execute_write(stmt)
        except Exception as exc:
            log.warning("Constraint may already exist: %s", exc)

    for stmt in INDEXES:
        try:
            conn.execute_write(stmt)
        except Exception as exc:
            log.warning("Index may already exist: %s", exc)

    log.info("Schema applied: %d constraints, %d indexes", len(CONSTRAINTS), len(INDEXES))
