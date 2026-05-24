"""
Graph builder — reads enriched JSON files and populates the Neo4j knowledge graph.

Produces a document-faithful hierarchy:

    Regulation
    ├── Recital …
    ├── Chapter
    │   ├── Section          (optional)
    │   │   └── Article
    │   │       ├── Paragraph          (top-level: "1", "2", "intro")
    │   │       │   └── SubParagraph   ("1(a)", "1(b)", "2(i)" …)
    │   │       └── Definition         (only on Art 3 — Definitions)
    │   │           └── SubParagraph   (sub-items of a definition)
    │   └── Article          (direct, without section wrapper)
    │       └── …
    └── Annex
        └── AnnexSection

Cross-regulation links (OVERLAPS, SIMILAR_TERM) are curated thematic pairs.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, ClassVar

from graph.connection import Neo4jConnection
from graph.schema import apply_schema

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_REG_META: dict[str, dict[str, str]] = {
    "ai_act": {
        "id": "eu_ai_act",
        "short_name": "EU AI Act",
        "title_formal": "Regulation (EU) 2024/1689",
        "celex": "32024R1689",
        "oj_reference": "OJ L, 2024/1689, 12.7.2024",
        "entry_into_force": "2024-08-01",
        "eur_lex_url": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng",
    },
    "dsa": {
        "id": "eu_dsa_act",
        "short_name": "DSA",
        "title_formal": "Regulation (EU) 2022/2065",
        "celex": "32022R2065",
        "oj_reference": "OJ L 277, 27.10.2022",
        "entry_into_force": "2022-11-16",
        "eur_lex_url": "https://eur-lex.europa.eu/eli/reg/2022/2065/oj/eng",
    },
}

# Roman-numeral markers used as sub-items inside definitions.
# Multi-char only (ii, iii, iv …) — single chars (i, v, x) overlap with letters.
_MULTI_ROMAN = frozenset(
    [
        "ii",
        "iii",
        "iv",
        "vi",
        "vii",
        "viii",
        "ix",
    ]
)
# Single-char markers ambiguous between letter and roman numeral.
_AMBIGUOUS_SINGLE = frozenset(["i", "v", "x"])


class GraphBuilder:
    """Reads enriched JSON and writes nodes/edges in batched UNWIND queries."""

    def __init__(self, conn: Neo4jConnection) -> None:
        self.conn = conn
        self._counters: dict[str, int] = {}

    # -- public entry points --------------------------------------------------

    def ingest_all(self, data_dir: Path | None = None) -> dict[str, Any]:
        """Ingest all *_enriched.json files found in *data_dir*."""
        data_dir = data_dir or DATA_DIR
        apply_schema(self.conn)
        results: dict[str, Any] = {}
        for path in sorted(data_dir.glob("*_enriched.json")):
            log.info("Ingesting %s ...", path.name)
            stats = self._ingest_file(path)
            results[path.stem] = stats

        # After all files are ingested, build cross-regulation relationships
        if len(results) > 1:
            cross_stats = self._build_cross_regulation_links()
            results["cross_regulation"] = cross_stats

        return results

    def ingest_file(self, path: Path) -> dict[str, Any]:
        apply_schema(self.conn)
        return self._ingest_file(path)

    def clear_graph(self) -> None:
        self.conn.execute_write("MATCH (n) DETACH DELETE n")
        log.warning("Graph cleared")

    # -- private pipeline -----------------------------------------------------

    def _ingest_file(self, path: Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        doc_type = data["document_type"]
        reg_meta = _REG_META.get(doc_type, {})
        reg_id = reg_meta.get("id", doc_type)

        self._counters = {
            "regulations": 0,
            "chapters": 0,
            "sections": 0,
            "articles": 0,
            "paragraphs": 0,
            "subparagraphs": 0,
            "recitals": 0,
            "annexes": 0,
            "definitions": 0,
            "references": 0,
        }

        # 1. Regulation
        self._create_regulation(data, reg_meta)
        log.info("  Regulation node created")

        # 2. Recitals (batched)
        self._batch_recitals(data.get("recitals", []), reg_id)
        log.info("  %d recitals", self._counters["recitals"])

        # 3. Chapters -> Sections -> Articles -> Paragraphs -> SubParagraphs (batched)
        self._batch_structural(data.get("chapters", []), reg_id)
        log.info(
            "  %d chapters, %d sections, %d articles, %d paragraphs, %d subparagraphs",
            self._counters["chapters"],
            self._counters["sections"],
            self._counters["articles"],
            self._counters["paragraphs"],
            self._counters["subparagraphs"],
        )

        # 4. Annexes (batched)
        self._batch_annexes(data.get("annexes", []), reg_id)
        log.info("  %d annexes", self._counters["annexes"])

        # 5. Definitions from Article 3
        self._batch_definitions(data.get("chapters", []), reg_id)
        log.info("  %d definitions", self._counters["definitions"])

        # 6. Cross-references
        self._batch_link_references(data.get("all_links", []), reg_id)
        self._batch_metadata_references(data.get("chapters", []), reg_id)

        # 7. Recital → Article references (parsed from recital text)
        self._batch_recital_article_references(data.get("recitals", []), reg_id)
        log.info("  %d references", self._counters["references"])

        log.info("Ingestion complete for %s: %s", doc_type, self._counters)
        return dict(self._counters)

    # -- Regulation (single node) ---------------------------------------------

    def _create_regulation(self, data: dict, reg_meta: dict) -> None:
        stats = data.get("stats", {})
        self.conn.execute_write(
            """
            MERGE (r:Regulation {id: $id})
            SET r += $props
            """,
            {
                "id": reg_meta.get("id", data["document_type"]),
                "props": {
                    "title": data.get("title", ""),
                    "short_name": reg_meta.get("short_name", ""),
                    "title_formal": reg_meta.get("title_formal", ""),
                    "celex": reg_meta.get("celex", data.get("celex_number", "")),
                    "document_type": data["document_type"],
                    "oj_reference": reg_meta.get("oj_reference", ""),
                    "entry_into_force": reg_meta.get("entry_into_force", ""),
                    "eur_lex_url": reg_meta.get("eur_lex_url", ""),
                    "recital_count": stats.get("recital_count", 0),
                    "chapter_count": stats.get("chapter_count", 0),
                    "article_count": stats.get("article_count", 0),
                    "annex_count": stats.get("annex_count", 0),
                },
            },
        )
        self._counters["regulations"] += 1

    # -- Recitals (batched) ---------------------------------------------------

    def _batch_recitals(self, recitals: list[dict], reg_id: str) -> None:
        if not recitals:
            return
        rows = []
        for rec in recitals:
            num = rec["number"]
            rows.append(
                {
                    "id": f"{reg_id}_rec_{num}",
                    "number": int(num) if num.isdigit() else num,
                    "text": rec.get("text", ""),
                    "regulation": reg_id,
                }
            )

        # Create all recital nodes + link to regulation
        self.conn.execute_write(
            """
            UNWIND $rows AS row
            MERGE (rec:Recital {id: row.id})
            SET rec.number     = row.number,
                rec.text       = row.text,
                rec.regulation = row.regulation
            WITH rec, row
            MATCH (r:Regulation {id: row.regulation})
            MERGE (r)-[:CONTAINS]->(rec)
            MERGE (rec)-[:PART_OF]->(r)
            """,
            {"rows": rows},
        )

        # Sequential NEXT/PREVIOUS edges
        pairs = [{"a": rows[i]["id"], "b": rows[i + 1]["id"]} for i in range(len(rows) - 1)]
        if pairs:
            self.conn.execute_write(
                """
                UNWIND $pairs AS p
                MATCH (a:Recital {id: p.a}), (b:Recital {id: p.b})
                MERGE (a)-[:NEXT]->(b)
                MERGE (b)-[:PREVIOUS]->(a)
                """,
                {"pairs": pairs},
            )
        self._counters["recitals"] += len(rows)

    # -- Structural: Chapters -> Sections -> Articles -> Paragraphs ----------

    def _batch_structural(self, chapters: list[dict], reg_id: str) -> None:
        ch_rows: list[dict] = []
        sec_rows: list[dict] = []
        art_rows: list[dict] = []
        par_rows: list[dict] = []
        subpar_rows: list[dict] = []
        subpar_parent_links: list[dict] = []
        subpar_article_links: list[dict] = []

        ch_pairs: list[dict] = []
        sec_pairs: list[dict] = []
        art_pairs: list[dict] = []
        par_pairs: list[dict] = []

        prev_ch_id = None
        for ch in chapters:
            ch_num = ch["number"]
            ch_id = f"{reg_id}_ch_{ch_num}"
            meta = ch.get("metadata", {})

            ch_rows.append(
                {
                    "id": ch_id,
                    "number": ch_num,
                    "title": ch.get("title", ""),
                    "regulation": reg_id,
                    "summary": meta.get("summary", ""),
                    "key_themes": meta.get("key_themes", []),
                    "scope_description": meta.get("scope_description", ""),
                    "regulatory_functions": meta.get("regulatory_functions", []),
                    "article_count": meta.get("article_count", 0),
                }
            )

            if prev_ch_id:
                ch_pairs.append({"a": prev_ch_id, "b": ch_id})
            prev_ch_id = ch_id

            # Articles directly under chapter
            self._collect_articles(
                ch.get("articles", []),
                ch_id,
                ch_num,
                None,
                reg_id,
                art_rows,
                par_rows,
                subpar_rows,
                subpar_parent_links,
                subpar_article_links,
                art_pairs,
                par_pairs,
            )

            # Sections
            prev_sec_id = None
            for sec in ch.get("sections", []):
                sec_num = sec["number"]
                sec_id = f"{ch_id}_sec_{sec_num}"
                smeta = sec.get("metadata", {})

                sec_rows.append(
                    {
                        "id": sec_id,
                        "number": sec_num,
                        "title": sec.get("title", ""),
                        "regulation": reg_id,
                        "chapter": ch_num,
                        "key_themes": smeta.get("key_themes", []),
                        "applies_to": smeta.get("applies_to", []),
                        "article_count": smeta.get("article_count", 0),
                        "parent_id": ch_id,
                    }
                )

                if prev_sec_id:
                    sec_pairs.append({"a": prev_sec_id, "b": sec_id})
                prev_sec_id = sec_id

                self._collect_articles(
                    sec.get("articles", []),
                    sec_id,
                    ch_num,
                    sec_num,
                    reg_id,
                    art_rows,
                    par_rows,
                    subpar_rows,
                    subpar_parent_links,
                    subpar_article_links,
                    art_pairs,
                    par_pairs,
                )

        # -- Execute batch writes --

        # Chapters
        if ch_rows:
            self.conn.execute_write(
                """
                UNWIND $rows AS row
                MERGE (c:Chapter {id: row.id})
                SET c.number              = row.number,
                    c.title               = row.title,
                    c.regulation          = row.regulation,
                    c.summary             = row.summary,
                    c.key_themes          = row.key_themes,
                    c.scope_description   = row.scope_description,
                    c.regulatory_functions = row.regulatory_functions,
                    c.article_count       = row.article_count
                WITH c, row
                MATCH (r:Regulation {id: row.regulation})
                MERGE (r)-[:CONTAINS]->(c)
                MERGE (c)-[:PART_OF]->(r)
                """,
                {"rows": ch_rows},
            )
            self._counters["chapters"] += len(ch_rows)

        if ch_pairs:
            self.conn.execute_write(
                """
                UNWIND $pairs AS p
                MATCH (a:Chapter {id: p.a}), (b:Chapter {id: p.b})
                MERGE (a)-[:NEXT]->(b)
                MERGE (b)-[:PREVIOUS]->(a)
                """,
                {"pairs": ch_pairs},
            )

        # Sections
        if sec_rows:
            self.conn.execute_write(
                """
                UNWIND $rows AS row
                MERGE (s:Section {id: row.id})
                SET s.number        = row.number,
                    s.title         = row.title,
                    s.regulation    = row.regulation,
                    s.chapter       = row.chapter,
                    s.key_themes    = row.key_themes,
                    s.applies_to   = row.applies_to,
                    s.article_count = row.article_count
                WITH s, row
                MATCH (c:Chapter {id: row.parent_id})
                MERGE (c)-[:CONTAINS]->(s)
                MERGE (s)-[:PART_OF]->(c)
                """,
                {"rows": sec_rows},
            )
            self._counters["sections"] += len(sec_rows)

        if sec_pairs:
            self.conn.execute_write(
                """
                UNWIND $pairs AS p
                MATCH (a:Section {id: p.a}), (b:Section {id: p.b})
                MERGE (a)-[:NEXT]->(b)
                MERGE (b)-[:PREVIOUS]->(a)
                """,
                {"pairs": sec_pairs},
            )

        # Articles
        if art_rows:
            self.conn.execute_write(
                """
                UNWIND $rows AS row
                MERGE (a:Article {id: row.id})
                SET a.number            = row.number,
                    a.title             = row.title,
                    a.regulation        = row.regulation,
                    a.chapter           = row.chapter,
                    a.section           = row.section,
                    a.full_text         = row.full_text,
                    a.summary           = row.summary,
                    a.key_obligations   = row.key_obligations,
                    a.key_topics        = row.key_topics,
                    a.applies_to        = row.applies_to,
                    a.cross_references  = row.cross_references,
                    a.regulatory_action = row.regulatory_action,
                    a.paragraph_count   = row.paragraph_count
                """,
                {"rows": art_rows},
            )
            # Link articles to parents (separate to handle mixed parent types)
            self.conn.execute_write(
                """
                UNWIND $rows AS row
                MATCH (a:Article {id: row.id})
                MATCH (p {id: row.parent_id})
                MERGE (p)-[:CONTAINS]->(a)
                MERGE (a)-[:PART_OF]->(p)
                """,
                {"rows": art_rows},
            )
            self._counters["articles"] += len(art_rows)

        if art_pairs:
            self.conn.execute_write(
                """
                UNWIND $pairs AS p
                MATCH (a:Article {id: p.a}), (b:Article {id: p.b})
                MERGE (a)-[:NEXT]->(b)
                MERGE (b)-[:PREVIOUS]->(a)
                """,
                {"pairs": art_pairs},
            )

        # Paragraphs (top-level only: "1", "2", "intro")
        if par_rows:
            self.conn.execute_write(
                """
                UNWIND $rows AS row
                MERGE (p:Paragraph {id: row.id})
                SET p.number     = row.number,
                    p.text       = row.text,
                    p.regulation = row.regulation
                WITH p, row
                MATCH (a:Article {id: row.parent_id})
                MERGE (a)-[:CONTAINS]->(p)
                MERGE (p)-[:PART_OF]->(a)
                """,
                {"rows": par_rows},
            )
            self._counters["paragraphs"] += len(par_rows)

        if par_pairs:
            self.conn.execute_write(
                """
                UNWIND $pairs AS p
                MATCH (a:Paragraph {id: p.a}), (b:Paragraph {id: p.b})
                MERGE (a)-[:NEXT]->(b)
                MERGE (b)-[:PREVIOUS]->(a)
                """,
                {"pairs": par_pairs},
            )

        # SubParagraphs ("1(a)", "2(b)", "3(i)" …)
        if subpar_rows:
            self.conn.execute_write(
                """
                UNWIND $rows AS row
                MERGE (sp:SubParagraph {id: row.id})
                SET sp.number     = row.number,
                    sp.text       = row.text,
                    sp.regulation = row.regulation
                """,
                {"rows": subpar_rows},
            )
            self._counters["subparagraphs"] += len(subpar_rows)

        # Link SubParagraphs to parent Paragraphs
        if subpar_parent_links:
            self.conn.execute_write(
                """
                UNWIND $links AS link
                MATCH (child:SubParagraph {id: link.child})
                MATCH (parent:Paragraph {id: link.parent})
                MERGE (parent)-[:CONTAINS]->(child)
                MERGE (child)-[:PART_OF]->(parent)
                """,
                {"links": subpar_parent_links},
            )

        # Fallback: SubParagraphs whose parent Paragraph doesn't exist → link to Article
        if subpar_article_links:
            self.conn.execute_write(
                """
                UNWIND $links AS link
                MATCH (child:SubParagraph {id: link.child})
                WHERE NOT (child)-[:PART_OF]->(:Paragraph)
                MATCH (art:Article {id: link.article})
                MERGE (art)-[:CONTAINS]->(child)
                MERGE (child)-[:PART_OF]->(art)
                """,
                {"links": subpar_article_links},
            )

    def _collect_articles(
        self,
        articles: list[dict],
        parent_id: str,
        ch_num: str,
        sec_num: str | None,
        reg_id: str,
        art_rows: list,
        par_rows: list,
        subpar_rows: list,
        subpar_parent_links: list,
        subpar_article_links: list,
        art_pairs: list,
        par_pairs: list,
    ) -> None:
        """Collect article, paragraph, and sub-paragraph data into batch lists.

        Top-level paragraphs (plain numbers, "intro") become Paragraph nodes.
        Sub-items like 1(a), 2(i) become SubParagraph nodes linked to their parent.
        Definition-article paragraphs (0(N)) are skipped — handled by _batch_definitions.
        """
        prev_art_id = None
        for art in articles:
            art_num = art["number"]
            art_id = f"{reg_id}_art_{art_num}"
            meta = art.get("metadata", {})
            full_text = " ".join(p.get("text", "") for p in art.get("paragraphs", []))

            art_rows.append(
                {
                    "id": art_id,
                    "number": int(art_num) if art_num.isdigit() else art_num,
                    "title": art.get("title", ""),
                    "regulation": reg_id,
                    "chapter": ch_num,
                    "section": sec_num,
                    "full_text": full_text,
                    "summary": meta.get("summary", ""),
                    "key_obligations": meta.get("key_obligations", []),
                    "key_topics": meta.get("key_topics", []),
                    "applies_to": meta.get("applies_to", []),
                    "cross_references": meta.get("cross_references", []),
                    "regulatory_action": meta.get("regulatory_action", []),
                    "paragraph_count": meta.get("paragraph_count", 0),
                    "parent_id": parent_id,
                }
            )

            if prev_art_id:
                art_pairs.append({"a": prev_art_id, "b": art_id})
            prev_art_id = art_id

            # Determine if this is the definitions article
            is_def_article = art_num == "3" and "definition" in art.get("title", "").lower()

            # Pre-scan: which plain-number paragraphs exist (for sub-par linking)
            existing_par_nums: set[str] = set()
            for par in art.get("paragraphs", []):
                raw_num = par["number"]
                if raw_num.isdigit() or raw_num == "intro":
                    existing_par_nums.add(raw_num)

            prev_par_id = None
            seen: set[str] = set()
            for par in art.get("paragraphs", []):
                raw_num = par["number"]
                if raw_num in seen:
                    continue
                seen.add(raw_num)
                text = par.get("text", "")

                # Definition-article paragraphs 0(…) → handled by _batch_definitions
                if is_def_article and raw_num.startswith("0("):
                    continue

                # "intro" and plain digits → top-level Paragraph
                if raw_num == "intro" or raw_num.isdigit():
                    par_id = f"{art_id}_par_{raw_num}"
                    par_rows.append(
                        {
                            "id": par_id,
                            "number": raw_num,
                            "text": text,
                            "regulation": reg_id,
                            "parent_id": art_id,
                        }
                    )
                    if prev_par_id:
                        par_pairs.append({"a": prev_par_id, "b": par_id})
                    prev_par_id = par_id
                    continue

                # Sub-paragraph: N(letter), N(roman), N(digit) → SubParagraph
                point_match = re.match(r"^(\d+)\(([a-z]+|\d+)\)$", raw_num)
                if point_match:
                    base_num = point_match.group(1)
                    subpar_id = f"{art_id}_subpar_{raw_num}"
                    subpar_rows.append(
                        {
                            "id": subpar_id,
                            "number": raw_num,
                            "text": text,
                            "regulation": reg_id,
                        }
                    )
                    # Try linking to parent Paragraph
                    parent_par_id = f"{art_id}_par_{base_num}"
                    subpar_parent_links.append(
                        {
                            "child": subpar_id,
                            "parent": parent_par_id,
                        }
                    )
                    # Fallback if parent paragraph doesn't exist in data
                    if base_num not in existing_par_nums:
                        subpar_article_links.append(
                            {
                                "child": subpar_id,
                                "article": art_id,
                            }
                        )

    # -- Annexes (batched) ----------------------------------------------------

    def _batch_annexes(self, annexes: list[dict], reg_id: str) -> None:
        if not annexes:
            return
        anx_rows = []
        anx_pairs = []
        sec_rows = []
        prev_anx_id = None

        for anx in annexes:
            anx_num = anx["number"]
            anx_id = f"{reg_id}_anx_{anx_num}"
            content = anx.get("content", "")

            anx_rows.append(
                {
                    "id": anx_id,
                    "number": anx_num,
                    "title": anx.get("title", ""),
                    "content": content,
                    "regulation": reg_id,
                }
            )

            if prev_anx_id:
                anx_pairs.append({"a": prev_anx_id, "b": anx_id})
            prev_anx_id = anx_id

            # Parse sections from content
            for match in re.finditer(
                r"Section\s+([A-Z])\.\s*(.+?)(?=Section\s+[A-Z]\.|$)",
                content,
                re.DOTALL,
            ):
                sec_rows.append(
                    {
                        "id": f"{anx_id}_sec_{match.group(1)}",
                        "number": match.group(1),
                        "title": f"Section {match.group(1)}",
                        "content": match.group(2).strip()[:2000],
                        "regulation": reg_id,
                        "parent_id": anx_id,
                    }
                )

        self.conn.execute_write(
            """
            UNWIND $rows AS row
            MERGE (a:Annex {id: row.id})
            SET a.number     = row.number,
                a.title      = row.title,
                a.content    = row.content,
                a.regulation = row.regulation
            WITH a, row
            MATCH (r:Regulation {id: row.regulation})
            MERGE (r)-[:CONTAINS]->(a)
            MERGE (a)-[:PART_OF]->(r)
            """,
            {"rows": anx_rows},
        )
        self._counters["annexes"] += len(anx_rows)

        if anx_pairs:
            self.conn.execute_write(
                """
                UNWIND $pairs AS p
                MATCH (a:Annex {id: p.a}), (b:Annex {id: p.b})
                MERGE (a)-[:NEXT]->(b)
                MERGE (b)-[:PREVIOUS]->(a)
                """,
                {"pairs": anx_pairs},
            )

        if sec_rows:
            self.conn.execute_write(
                """
                UNWIND $rows AS row
                MERGE (s:AnnexSection {id: row.id})
                SET s.number     = row.number,
                    s.title      = row.title,
                    s.content    = row.content,
                    s.regulation = row.regulation
                WITH s, row
                MATCH (a:Annex {id: row.parent_id})
                MERGE (a)-[:CONTAINS]->(s)
                MERGE (s)-[:PART_OF]->(a)
                """,
                {"rows": sec_rows},
            )

    # -- Definitions from definitions article (batched) -----------------------

    def _batch_definitions(self, chapters: list[dict], reg_id: str) -> None:
        """Build Definition nodes as children of the definitions article.

        Structure produced:
            Article 3 -[CONTAINS]-> Definition  (for each defined term)
            Definition -[CONTAINS]-> SubParagraph  (for multi-part definitions)

        Handles:
          * AI Act: paragraphs numbered 0(1), 0(2) … with 'term' means …
          * DSA:   paragraphs numbered 0(a), 0(b) … with 'term' means …
                   plus roman-numeral sub-definitions 0(i), 0(ii), 0(iii)
        """
        def_article = self._find_definitions_article(chapters)
        if not def_article:
            return

        art_id = f"{reg_id}_art_3"

        term_pattern = re.compile(
            r"['\u2018\u201C](.+?)['\u2019\u201D]\s+means\s+(.+)",
            re.IGNORECASE,
        )

        # Separate main definitions from sub-definitions.
        # Main definitions match the 'term' means ... pattern.
        # Sub-definitions (e.g. roman numeral sub-items of a parent def) don't.
        def_pars: list[dict] = []
        subdef_pars: list[dict] = []

        for par in def_article.get("paragraphs", []):
            raw_num = par["number"]
            text = par.get("text", "")

            if raw_num == "intro" or text.strip() in ("\u2014", "-", ""):
                continue
            if not raw_num.startswith("0("):
                continue

            marker = raw_num[2:-1]  # extract part inside 0(…)
            matches_pattern = bool(term_pattern.match(text))

            if matches_pattern:
                # Clearly a main definition ('term' means …)
                def_pars.append(par)
            elif marker in _MULTI_ROMAN:
                # Multi-char roman numeral (ii, iii, iv …) → sub-definition
                subdef_pars.append(par)
            elif marker in _AMBIGUOUS_SINGLE and not matches_pattern:
                # Single-char ambiguous (i, v, x) that doesn't match pattern → sub-def
                subdef_pars.append(par)
            else:
                # Doesn't match pattern but not a roman numeral → fallback definition
                def_pars.append(par)

        # Build Definition rows
        def_rows: list[dict] = []
        for par in def_pars:
            raw_num = par["number"]
            text = par.get("text", "")

            term_match = term_pattern.match(text)
            if term_match:
                term = term_match.group(1).strip()
                definition_text = term_match.group(2).strip()
            else:
                term = f"definition_{raw_num}"
                definition_text = text

            safe_term = re.sub(r"[^a-z0-9_]", "_", term.lower())
            def_rows.append(
                {
                    "id": f"{reg_id}_def_{safe_term}",
                    "term": term,
                    "definition_text": definition_text,
                    "full_text": text,
                    "regulation": reg_id,
                    "paragraph_num": raw_num,
                }
            )

        if not def_rows:
            return

        # Create definition nodes as children of the Article
        self.conn.execute_write(
            """
            UNWIND $rows AS row
            MERGE (d:Definition {id: row.id})
            SET d.term            = row.term,
                d.definition_text = row.definition_text,
                d.full_text       = row.full_text,
                d.regulation      = row.regulation,
                d.paragraph_num   = row.paragraph_num
            WITH d, row
            MATCH (a:Article {id: $art_id})
            MERGE (a)-[:CONTAINS]->(d)
            MERGE (d)-[:PART_OF]->(a)
            """,
            {"rows": def_rows, "art_id": art_id},
        )

        # Sequential NEXT/PREVIOUS among definitions
        pairs = [
            {"a": def_rows[i]["id"], "b": def_rows[i + 1]["id"]} for i in range(len(def_rows) - 1)
        ]
        if pairs:
            self.conn.execute_write(
                """
                UNWIND $pairs AS p
                MATCH (a:Definition {id: p.a}), (b:Definition {id: p.b})
                MERGE (a)-[:NEXT]->(b)
                MERGE (b)-[:PREVIOUS]->(a)
                """,
                {"pairs": pairs},
            )

        self._counters["definitions"] += len(def_rows)

        # Sub-definition items (roman numerals) → SubParagraph children of parent Definition
        if subdef_pars:
            # Find the parent definition that introduces sub-items.
            # This is the definition whose text contains phrasing like
            # "the following" or "consists of", indicating sub-items follow.
            parent_with_subitems = None
            for row in def_rows:
                text_lower = row.get("full_text", "").lower()
                if "the following" in text_lower or "consists of" in text_lower:
                    parent_with_subitems = row
                    break

            subdef_rows: list[dict] = []
            for spar in subdef_pars:
                raw_num = spar["number"]
                text = spar.get("text", "")
                subpar_id = f"{art_id}_subpar_{raw_num}"

                parent_def_id = parent_with_subitems["id"] if parent_with_subitems else art_id

                subdef_rows.append(
                    {
                        "id": subpar_id,
                        "number": raw_num,
                        "text": text,
                        "regulation": reg_id,
                        "parent_id": parent_def_id,
                    }
                )

            if subdef_rows:
                self.conn.execute_write(
                    """
                    UNWIND $rows AS row
                    MERGE (sp:SubParagraph {id: row.id})
                    SET sp.number     = row.number,
                        sp.text       = row.text,
                        sp.regulation = row.regulation
                    WITH sp, row
                    MATCH (d:Definition {id: row.parent_id})
                    MERGE (d)-[:CONTAINS]->(sp)
                    MERGE (sp)-[:PART_OF]->(d)
                    """,
                    {"rows": subdef_rows},
                )
                self._counters["subparagraphs"] += len(subdef_rows)

    def _find_definitions_article(self, chapters: list[dict]) -> dict | None:
        """Locate the definitions article (usually Article 3)."""
        for ch in chapters:
            for art in ch.get("articles", []):
                if art.get("number") == "3" and "definition" in art.get("title", "").lower():
                    return art
            for sec in ch.get("sections", []):
                for art in sec.get("articles", []):
                    if art.get("number") == "3" and "definition" in art.get("title", "").lower():
                        return art
        for ch in chapters:
            for art in ch.get("articles", []):
                if art.get("number") == "3":
                    return art
            for sec in ch.get("sections", []):
                for art in sec.get("articles", []):
                    if art.get("number") == "3":
                        return art
        return None

    # -- Cross-references from all_links (batched) ----------------------------

    def _batch_link_references(self, links: list[dict], reg_id: str) -> None:
        ref_rows = []
        ext_rows = []

        for link in links:
            source_type = link.get("source_type", "")
            source_id_raw = link.get("source_id", "")
            link_type = link.get("link_type", "")
            target_url = link.get("target_url", "")
            anchor = link.get("anchor_text", "")

            if source_type == "recital":
                src = f"{reg_id}_rec_{re.sub(r'^rec_', '', source_id_raw)}"
            elif source_type == "article":
                src = f"{reg_id}_art_{re.sub(r'^art_', '', source_id_raw)}"
            elif source_type == "annex":
                src = f"{reg_id}_anx_{re.sub(r'^anx_', '', source_id_raw)}"
            else:
                continue

            if link_type == "internal_other":
                art_match = re.search(r"Article\s+(\d+)", anchor)
                if art_match:
                    ref_rows.append(
                        {
                            "src": src,
                            "tgt": f"{reg_id}_art_{art_match.group(1)}",
                            "anchor": anchor,
                            "url": target_url,
                        }
                    )
            elif link_type in ("external_eli", "external_other"):
                ext_rows.append(
                    {
                        "src": src,
                        "ref": f"{anchor} -> {target_url}",
                    }
                )

        if ref_rows:
            self.conn.execute_write(
                """
                UNWIND $rows AS row
                MATCH (src {id: row.src}), (tgt {id: row.tgt})
                MERGE (src)-[:REFERENCES {anchor: row.anchor, url: row.url}]->(tgt)
                """,
                {"rows": ref_rows},
            )
            self._counters["references"] += len(ref_rows)

        if ext_rows:
            self.conn.execute_write(
                """
                UNWIND $rows AS row
                MATCH (src {id: row.src})
                SET src.external_references = coalesce(src.external_references, []) + row.ref
                """,
                {"rows": ext_rows},
            )

    # -- Cross-references from article metadata (batched) ---------------------

    def _batch_metadata_references(self, chapters: list[dict], reg_id: str) -> None:
        ref_rows = []
        for ch in chapters:
            all_articles = list(ch.get("articles", []))
            for sec in ch.get("sections", []):
                all_articles.extend(sec.get("articles", []))

            for art in all_articles:
                art_id = f"{reg_id}_art_{art['number']}"
                for ref_text in art.get("metadata", {}).get("cross_references", []):
                    art_ref = re.match(r"Article\s+(\d+)", ref_text)
                    anx_ref = re.match(r"Annex\s+([IVXLCDM]+)", ref_text)
                    ch_ref = re.match(r"Chapter\s+([IVXLCDM]+)", ref_text)

                    target_id = None
                    if art_ref:
                        target_id = f"{reg_id}_art_{art_ref.group(1)}"
                    elif anx_ref:
                        target_id = f"{reg_id}_anx_{anx_ref.group(1)}"
                    elif ch_ref:
                        target_id = f"{reg_id}_ch_{ch_ref.group(1)}"

                    if target_id:
                        ref_rows.append(
                            {
                                "src": art_id,
                                "tgt": target_id,
                                "context": ref_text,
                            }
                        )

        if ref_rows:
            self.conn.execute_write(
                """
                UNWIND $rows AS row
                MATCH (src:Article {id: row.src}), (tgt {id: row.tgt})
                MERGE (src)-[:REFERENCES {context: row.context}]->(tgt)
                """,
                {"rows": ref_rows},
            )
            self._counters["references"] += len(ref_rows)

    # -- Cross-regulation relationships ---------------------------------------

    # Curated thematic overlap map between EU AI Act and DSA.
    # Each entry: (theme, [(ai_act_article_number, dsa_article_number), ...])
    _CURATED_OVERLAPS: ClassVar[list[tuple[str, list[tuple[str, str]]]]] = [
        (
            "scope_and_definitions",
            [
                ("1", "1"),  # Subject matter
                ("2", "2"),  # Scope
                ("3", "3"),  # Definitions
            ],
        ),
        (
            "transparency",
            [
                ("13", "14"),  # Transparency / Terms of service
                ("50", "26"),  # AI-generated content / Online advertising
                ("50", "27"),  # AI output labelling / Recommender systems
                ("13", "38"),  # Transparency / Additional VLOP ad transparency
                ("13", "39"),  # Transparency / Access to data for scrutiny
            ],
        ),
        (
            "risk_management",
            [
                ("9", "34"),  # Risk management system / VLOP risk assessment
                ("9", "35"),  # Risk management system / VLOP risk mitigation
                ("51", "34"),  # Systemic-risk GPAI / VLOP systemic risk
                ("55", "35"),  # GPAI systemic risk obligations / VLOP risk mitigation
            ],
        ),
        (
            "provider_obligations",
            [
                (
                    "16",
                    "11",
                ),  # High-risk AI providers / Intermediary service obligations
                ("16", "13"),  # High-risk AI providers / Hosting provider obligations
                ("17", "12"),  # Quality management / Points of contact
                ("16", "16"),  # Provider notification duties
            ],
        ),
        (
            "user_rights_and_remedies",
            [
                ("86", "17"),  # Right to explanation / Statement of reasons
                ("85", "20"),  # Right to lodge complaint / Internal complaints
                ("85", "21"),  # Right to lodge complaint / Out-of-court disputes
            ],
        ),
        (
            "enforcement_and_governance",
            [
                ("64", "49"),  # AI Board / Digital Services Coordinators
                ("65", "50"),  # Advisory forum / DSC requirements
                ("72", "51"),  # Monitoring and enforcement / Joint investigations
                ("99", "52"),  # Fines / Penalties
                ("74", "49"),  # Market surveillance / Competent authorities
                ("75", "56"),  # Mutual assistance / Cross-border cooperation
            ],
        ),
        (
            "data_and_profiling",
            [
                ("10", "26"),  # Data governance / Profiling restrictions
                ("10", "31"),  # Data governance / Data access for researchers
                ("14", "27"),  # Human oversight / Recommender systems
            ],
        ),
        (
            "prohibited_and_restricted_practices",
            [
                ("5", "14"),  # Prohibited AI practices / Content moderation ToS
                ("5", "25"),  # Prohibited practices / Dark-pattern restrictions
            ],
        ),
        (
            "reporting_and_recordkeeping",
            [
                ("62", "15"),  # Reporting / Transparency reporting
                ("12", "24"),  # Record-keeping / Platform transparency reporting
                ("62", "42"),  # Reporting / VLOP reporting obligations
            ],
        ),
    ]

    # Curated semantic definition pairs (AI Act term, DSA term, relation)
    _CURATED_DEF_PAIRS: ClassVar[list[tuple[str, str, str]]] = [
        ("provider", "intermediary service", "both define service providers"),
        ("profiling", "recommender system", "automated profiling systems"),
        ("deployer", "recipient of the service", "users of services"),
        ("systemic risk", "illegal content", "risk categories"),
    ]

    def _build_cross_regulation_links(self) -> dict[str, int]:
        """Build relationships between nodes of different regulations.

        Uses curated thematic mappings to create meaningful cross-regulation
        connections rather than broad keyword matching.

        Creates:
          * Regulation -[:RELATED_REGULATION]-> Regulation
          * Article -[:OVERLAPS {theme}]-> Article  (curated thematic overlap)
          * Definition -[:SIMILAR_TERM {reason}]-> Definition
        """
        counters = {"reg_links": 0, "article_overlaps": 0, "def_links": 0}

        # 1. Link regulations to each other
        self.conn.execute_write("""
            MATCH (a:Regulation), (b:Regulation)
            WHERE a.id < b.id
            MERGE (a)-[:RELATED_REGULATION]->(b)
            MERGE (b)-[:RELATED_REGULATION]->(a)
            """)
        reg_count = self.conn.execute_read("MATCH (r:Regulation) RETURN count(r) AS c")
        n = reg_count[0]["c"] if reg_count else 0
        counters["reg_links"] = n * (n - 1) if n > 1 else 0

        # 2. Curated article overlaps
        # Build a lookup of available regulation IDs
        regs = self.conn.execute_read("MATCH (r:Regulation) RETURN r.id AS id")
        reg_ids = {r["id"] for r in regs}

        # Determine which reg is AI Act and which is DSA
        ai_reg = next((r for r in reg_ids if "ai" in r.lower()), None)
        dsa_reg = next((r for r in reg_ids if "dsa" in r.lower()), None)

        if ai_reg and dsa_reg:
            overlap_rows = []
            for theme, pairs in self._CURATED_OVERLAPS:
                for ai_art_num, dsa_art_num in pairs:
                    overlap_rows.append(
                        {
                            "src": f"{ai_reg}_art_{ai_art_num}",
                            "tgt": f"{dsa_reg}_art_{dsa_art_num}",
                            "theme": theme,
                        }
                    )

            if overlap_rows:
                self.conn.execute_write(
                    """
                    UNWIND $rows AS row
                    MATCH (a:Article {id: row.src}), (b:Article {id: row.tgt})
                    MERGE (a)-[:OVERLAPS {theme: row.theme}]->(b)
                    MERGE (b)-[:OVERLAPS {theme: row.theme}]->(a)
                    """,
                    {"rows": overlap_rows},
                )
                counters["article_overlaps"] = len(overlap_rows)

            # 3. Curated definition pairs
            def_rows = []
            for ai_term, dsa_term, reason in self._CURATED_DEF_PAIRS:
                ai_safe = re.sub(r"[^a-z0-9_]", "_", ai_term.lower())
                dsa_safe = re.sub(r"[^a-z0-9_]", "_", dsa_term.lower())
                def_rows.append(
                    {
                        "src": f"{ai_reg}_def_{ai_safe}",
                        "tgt": f"{dsa_reg}_def_{dsa_safe}",
                        "reason": reason,
                    }
                )

            if def_rows:
                self.conn.execute_write(
                    """
                    UNWIND $rows AS row
                    MATCH (a:Definition {id: row.src}), (b:Definition {id: row.tgt})
                    MERGE (a)-[:SIMILAR_TERM {reason: row.reason}]->(b)
                    MERGE (b)-[:SIMILAR_TERM {reason: row.reason}]->(a)
                    """,
                    {"rows": def_rows},
                )
                counters["def_links"] = len(def_rows)

        log.info("Cross-regulation links: %s", counters)
        return counters

    # -- Recital ↔ Article references (text-mined) ----------------------------

    def _batch_recital_article_references(self, recitals: list[dict], reg_id: str) -> None:
        """Parse Article mentions in recital text and create REFERENCES edges."""
        if not recitals:
            return
        ref_rows: list[dict] = []
        for rec in recitals:
            num = rec["number"]
            rec_id = f"{reg_id}_rec_{num}"
            text = rec.get("text", "")
            art_nums = set(re.findall(r"Article\s+(\d+)", text))
            for art_num in art_nums:
                ref_rows.append(
                    {
                        "src": rec_id,
                        "tgt": f"{reg_id}_art_{art_num}",
                        "context": f"Recital {num} references Article {art_num}",
                    }
                )
        if ref_rows:
            self.conn.execute_write(
                """
                UNWIND $rows AS row
                MATCH (src:Recital {id: row.src}), (tgt:Article {id: row.tgt})
                MERGE (src)-[:REFERENCES {context: row.context}]->(tgt)
                MERGE (tgt)-[:REFERENCES {context: row.context}]->(src)
                """,
                {"rows": ref_rows},
            )
            self._counters["references"] += len(ref_rows)
            log.info("  %d recital-article references", len(ref_rows))
