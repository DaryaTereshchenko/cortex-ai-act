"""
Concrete document parsers and factory.

Each parser subclasses :class:`BaseDocumentExtractor` and is associated with
a :class:`ParsingSchema`.  Most EUR-Lex documents share the same HTML
template, so the subclasses are intentionally thin — override only the
methods whose behaviour genuinely differs for a given regulation.

Usage::

    from scraping.parsers import create_extractor, extract_from_html

    # Factory — pick parser by document type
    doc = create_extractor("dsa", soup).extract()

    # Convenience wrapper (defaults to AI Act)
    doc = extract_from_html(html_string, document_type="ai_act")
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from .extractor import BaseDocumentExtractor
from .models import Chapter, Document, Section
from .schemas import (
    AI_ACT_SCHEMA,
    DSA_SCHEMA,
    ParsingSchema,
    get_schema,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OJ-Style Extractor (shared by AI Act, DSA, and other EUR-Lex documents)
# ---------------------------------------------------------------------------


class OJStyleExtractor(BaseDocumentExtractor):
    """
    Extractor for EUR-Lex documents using ``oj-`` prefixed CSS classes.

    These documents share the same HTML structure:
    - Chapters and sections use ``p.oj-ti-section-1``, distinguished by text
    - Titles are in adjacent ``div.eli-title`` sibling elements
    - Annexes use ``div.eli-container[id^='anx']``

    Subclasses only need to specify their schema; the parsing logic is shared.
    """

    # In oj-style documents, chapters and sections both use ``oj-ti-section-1``.
    # We differentiate by checking for keywords in text.

    def _is_chapter_heading(self, elem: Tag, text: str) -> bool:
        return "oj-ti-section-1" in elem.get("class", []) and "CHAPTER" in text.upper()

    def _is_section_heading(self, elem: Tag) -> bool:
        text = elem.get_text(strip=True).upper()
        return (
            "oj-ti-section-1" in elem.get("class", [])
            and "SECTION" in text
            and "CHAPTER" not in text
        )

    def _is_article_element(self, elem: Tag) -> bool:
        return "oj-ti-art" in elem.get("class", []) or elem.get("id", "").startswith("art")

    # Chapter/section titles live in a ``div.eli-title`` next sibling.

    def _parse_chapter_heading(self, elem: Tag, text: str, index: int) -> Chapter:
        chapter = super()._parse_chapter_heading(elem, text, index)
        title_div = elem.find_next_sibling("div", class_="eli-title")
        if title_div:
            chapter.title = title_div.get_text(strip=True)
        return chapter

    def _parse_section_heading(self, elem: Tag, text: str, index: int) -> Section:
        section = super()._parse_section_heading(elem, text, index)
        title_div = elem.find_next_sibling("div", class_="eli-title")
        if title_div:
            section.title = title_div.get_text(strip=True)
        return section


# ---------------------------------------------------------------------------
# AI Act Parser
# ---------------------------------------------------------------------------


class AIActExtractor(OJStyleExtractor):
    """
    Parser for the EU Artificial Intelligence Act (Regulation 2024/1689).

    Uses the shared OJ-style parsing logic.  Add AI-Act-specific
    adjustments here (e.g. risk-level tagging) if needed.
    """

    def __init__(self, soup: BeautifulSoup, base_url: str = "") -> None:
        super().__init__(soup, schema=AI_ACT_SCHEMA, base_url=base_url)


# ---------------------------------------------------------------------------
# DSA Parser
# ---------------------------------------------------------------------------


class DSAExtractor(OJStyleExtractor):
    """
    Parser for the Digital Services Act (Regulation 2022/2065).

    Uses the shared OJ-style parsing logic.  Add DSA-specific
    adjustments here if needed.
    """

    def __init__(self, soup: BeautifulSoup, base_url: str = "") -> None:
        super().__init__(soup, schema=DSA_SCHEMA, base_url=base_url)


# ---------------------------------------------------------------------------
# Parser Registry / Factory
# ---------------------------------------------------------------------------

_PARSER_REGISTRY: dict[str, type[BaseDocumentExtractor]] = {
    "ai_act": AIActExtractor,
    "dsa": DSAExtractor,
}


def register_parser(
    document_type: str,
    parser_cls: type[BaseDocumentExtractor],
    schema: ParsingSchema | None = None,
) -> None:
    """
    Register a new parser class (and optionally its schema).

    Args:
        document_type: Unique identifier for the document type.
        parser_cls: Subclass of :class:`BaseDocumentExtractor`.
        schema: If provided, also registers the schema in
            :data:`schemas.SCHEMA_REGISTRY`.
    """
    _PARSER_REGISTRY[document_type] = parser_cls
    if schema is not None:
        from .schemas import SCHEMA_REGISTRY

        SCHEMA_REGISTRY[document_type] = schema


def create_extractor(
    document_type: str,
    soup: BeautifulSoup,
    base_url: str = "",
) -> BaseDocumentExtractor:
    """
    Factory: instantiate the correct parser for *document_type*.

    Falls back to :class:`BaseDocumentExtractor` with the matching schema
    if no explicit subclass is registered.

    Args:
        document_type: Key such as ``"ai_act"`` or ``"dsa"``.
        soup: Parsed HTML.
        base_url: Base URL for resolving relative links.

    Returns:
        A ready-to-use extractor instance.
    """
    schema = get_schema(document_type)
    parser_cls = _PARSER_REGISTRY.get(document_type)

    if parser_cls is not None:
        try:
            return parser_cls(soup, base_url=base_url)
        except TypeError:
            # Subclass didn't override __init__ — pass schema explicitly
            return parser_cls(soup, schema=schema, base_url=base_url)

    # No explicit subclass — use base extractor with the schema
    return BaseDocumentExtractor(soup, schema=schema, base_url=base_url)


# ---------------------------------------------------------------------------
# Backward-compatible convenience functions
# ---------------------------------------------------------------------------

# Alias so existing imports ``from scraping.extractor import EURLexExtractor``
# (via __init__.py) keep working.
EURLexExtractor = AIActExtractor


def extract_from_html(
    html: str | BeautifulSoup,
    base_url: str = "",
    document_type: str = "ai_act",
) -> Document:
    """
    Extract structured data from EUR-Lex HTML.

    Args:
        html: Raw HTML string or parsed BeautifulSoup object.
        base_url: Base URL for resolving relative links.
        document_type: Document type key (default ``"ai_act"``).

    Returns:
        Parsed :class:`Document` object.
    """
    soup = BeautifulSoup(html, "lxml") if isinstance(html, str) else html

    extractor = create_extractor(document_type, soup, base_url=base_url)
    return extractor.extract()


def extract_from_url(url: str, document_type: str = "ai_act") -> Document:
    """
    Fetch and extract structured data from a EUR-Lex URL.

    Args:
        url: EUR-Lex document URL.
        document_type: Document type key (default ``"ai_act"``).

    Returns:
        Parsed :class:`Document` object.
    """
    from .downloading import get_html_content

    soup = get_html_content(url)
    return extract_from_html(soup, base_url=url, document_type=document_type)


# ---------------------------------------------------------------------------
# CLI Entry Point  (python -m scraping.parsers [ai_act|dsa|all])
# ---------------------------------------------------------------------------


def _cli_extract(document_type: str) -> None:
    """Run extraction for a single document type and save JSON."""
    from .downloading import get_html_content, get_urls

    urls = get_urls()
    schema = get_schema(document_type)

    url = urls[schema.url_key]
    logger.info("Fetching %s from %s ...", document_type, url)
    soup = get_html_content(url)

    logger.info("Extracting %s document structure...", document_type)
    doc = extract_from_html(soup, base_url=url, document_type=document_type)

    # Link-type breakdown
    link_counts: dict[str, int] = {}
    for link in doc.all_links:
        lt = link.link_type.value
        link_counts[lt] = link_counts.get(lt, 0) + 1
    logger.info("Link breakdown: %s", link_counts)

    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / schema.output_filename

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Saved %s extraction to %s", document_type, output_path)


if __name__ == "__main__":
    _service_root = Path(__file__).resolve().parent.parent
    if str(_service_root) not in sys.path:
        sys.path.insert(0, str(_service_root))

    from logging_config import configure_logging

    configure_logging(level=logging.DEBUG, log_file="extractor.log")

    # Parse CLI argument: ai_act, dsa, or all (default: all)
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    doc_types = ["ai_act", "dsa"] if target == "all" else [target]

    logger.info("Extracting documents: %s", doc_types)

    try:
        for dt in doc_types:
            _cli_extract(dt)
        logger.info("All extractions completed successfully")
    except Exception as e:
        logger.exception("Extraction failed: %s", e)
        sys.exit(1)
