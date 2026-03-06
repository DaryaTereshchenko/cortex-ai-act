"""
Parsing schemas for EUR-Lex documents.

Each schema defines the CSS selectors, regex patterns, and structural
configuration needed to parse a specific type of EU legislative document.
New document types can be added by creating a new ``ParsingSchema`` instance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsingSchema:
    """
    Declarative configuration for parsing a EUR-Lex HTML document.

    All CSS selectors and regex patterns that may vary between document types
    are parameterised here so that the base extractor stays generic.

    Attributes:
        document_type: Short identifier (e.g. ``"ai_act"``, ``"dsa"``).
        url_key: Key in ``links.json`` used by ``downloading.py``.
        output_filename: Default filename for the extracted JSON output.
        title_selector: CSS selector for the document title.
        recital_container_selector: CSS selector for recital containers.
        recital_text_selector: CSS selector for paragraphs inside recitals.
        recital_id_pattern: Regex to extract the recital number from the
            container ``id`` attribute.
        chapter_selector: CSS selector for chapter headings.
        chapter_keyword: Keyword that identifies a chapter heading (e.g.
            ``"CHAPTER"``).
        chapter_number_pattern: Regex to capture the chapter number.
        chapter_title_pattern: Regex to capture the chapter title text.
        section_selector: CSS selector for section headings.
        section_number_pattern: Regex to capture the section number.
        article_title_selector: CSS selector for article title elements.
        article_subtitle_selector: CSS selector for article subtitle elements.
        article_number_pattern: Regex to capture the article number.
        article_id_pattern: Regex to extract the article number from an
            element ``id`` attribute.
        paragraph_selector: CSS selector for paragraph/list-item elements.
        annex_container_selector: CSS selector for annex containers.
        annex_id_pattern: Regex to extract the annex number from the container
            ``id`` attribute.
        annex_title_selector: CSS selector for annex titles.
        structural_selector: Combined CSS selector that finds all structural
            elements (chapters, sections, articles) in document order.
    """

    document_type: str
    url_key: str
    output_filename: str = "extracted.json"

    # --- Title ---
    title_selector: str = "p.title-doc-first, p.doc-ti"

    # --- Recitals ---
    recital_container_selector: str = "div.eli-subdivision[id^='rct']"
    recital_text_selector: str = "p.normal"
    recital_id_pattern: str = r"rct[_-]?(\d+)"

    # --- Chapters ---
    chapter_selector: str = "p.ti-grseq-1"
    chapter_keyword: str = "CHAPTER"
    chapter_number_pattern: str = r"CHAPTER\s+([IVXLCDM]+|\d+)"
    chapter_title_pattern: str = r"CHAPTER\s+[IVXLCDM\d]+\s*(.+)"

    # --- Sections ---
    section_selector: str = "p.ti-section-1"
    section_number_pattern: str = r"Section\s+(\d+)"

    # --- Articles ---
    article_title_selector: str = "p.ti-art"
    article_subtitle_selector: str = "p.sti-art"
    article_number_pattern: str = r"Article\s+(\d+)"
    article_id_pattern: str = r"art[_-]?(\d+)"

    # --- Paragraphs ---
    paragraph_selector: str = "p.normal, p.li, table.li"

    # --- Annexes ---
    annex_container_selector: str = "div.eli-subdivision[id^='anx']"
    annex_id_pattern: str = r"anx[_-]?([IVXLCDM]+|\d+)"
    annex_title_selector: str = "p.ti-grseq-1, p.title"

    # --- Composite structural selector (document-order traversal) ---
    structural_selector: str = (
        "p.ti-grseq-1, p.ti-section-1, p.ti-art, div.eli-subdivision[id^='art']"
    )


# ---------------------------------------------------------------------------
# Pre-built schemas
# ---------------------------------------------------------------------------

# Common selectors for EUR-Lex documents using "oj-" prefixed classes.
# Both AI Act and DSA share this HTML structure.
_OJ_SELECTORS = dict(
    title_selector="p.oj-doc-ti",
    recital_container_selector="div.eli-subdivision[id^='rct']",
    recital_text_selector="p.oj-normal",
    # Chapters and sections BOTH use oj-ti-section-1, distinguished by text
    chapter_selector="p.oj-ti-section-1",
    chapter_keyword="CHAPTER",
    section_selector="p.oj-ti-section-1",
    section_number_pattern=r"SECTION\s+(\d+)",
    article_title_selector="p.oj-ti-art",
    article_subtitle_selector="p.oj-sti-art",
    paragraph_selector="p.oj-normal",
    # Annexes use eli-container class
    annex_container_selector="div.eli-container[id^='anx']",
    annex_title_selector="p.oj-doc-ti",
    structural_selector="p.oj-ti-section-1, p.oj-ti-art",
)

AI_ACT_SCHEMA = ParsingSchema(
    document_type="ai_act",
    url_key="AI_ACT_URL",
    output_filename="ai_act_extracted.json",
    **_OJ_SELECTORS,
)

DSA_SCHEMA = ParsingSchema(
    document_type="dsa",
    url_key="DSA_URL",
    output_filename="dsa_extracted.json",
    **_OJ_SELECTORS,
)

# Registry mapping document type names to schemas.
SCHEMA_REGISTRY: dict[str, ParsingSchema] = {
    "ai_act": AI_ACT_SCHEMA,
    "dsa": DSA_SCHEMA,
}


def get_schema(document_type: str) -> ParsingSchema:
    """
    Look up a parsing schema by document type.

    Args:
        document_type: Key such as ``"ai_act"`` or ``"dsa"``.

    Returns:
        The matching :class:`ParsingSchema`.

    Raises:
        KeyError: If no schema is registered for *document_type*.
    """
    try:
        return SCHEMA_REGISTRY[document_type]
    except KeyError:
        available = ", ".join(sorted(SCHEMA_REGISTRY))
        raise KeyError(
            f"Unknown document type {document_type!r}. Available types: {available}"
        ) from None
