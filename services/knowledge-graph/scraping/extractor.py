"""
EUR-Lex base document extractor.

Provides :class:`BaseDocumentExtractor`, a schema-driven parser that extracts
recitals, articles, chapters, sections, annexes, and cross-references from
EUR-Lex HTML.  Concrete subclasses (in ``parsers.py``) can override any
``_extract_*`` method to handle document-specific quirks without duplicating
the shared logic.

Backward-compatible aliases (``EURLexExtractor``, ``extract_from_html``,
``extract_from_url``) are kept in ``parsers.py``.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from .models import (
    Annex,
    Article,
    Chapter,
    CrossReference,
    Document,
    Paragraph,
    Recital,
    Section,
    classify_link,
)
from .schemas import ParsingSchema

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Extractor (Template-Method pattern)
# ---------------------------------------------------------------------------


class BaseDocumentExtractor:
    """
    Schema-driven extractor for EUR-Lex HTML documents.

    Override any ``_extract_*`` / ``_parse_*`` hook to customise behaviour
    for a particular regulation without duplicating shared parsing logic.
    """

    def __init__(
        self,
        soup: BeautifulSoup,
        schema: ParsingSchema,
        base_url: str = "",
    ) -> None:
        self.soup = soup
        self.schema = schema
        self.base_url = base_url
        self._current_chapter: str | None = None
        self._current_section: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> Document:
        """Run the full extraction pipeline and return a :class:`Document`."""
        logger.info(
            "Starting extraction for document type: %s",
            self.schema.document_type,
        )

        doc = Document(
            title=self._extract_title(),
            celex_number=self._extract_celex(),
            document_type=self.schema.document_type,
        )

        doc.recitals = self._extract_recitals()
        logger.info("Extracted %d recitals", len(doc.recitals))

        doc.chapters, doc.articles = self._extract_body()
        logger.info(
            "Extracted %d chapters, %d standalone articles",
            len(doc.chapters),
            len(doc.articles),
        )

        doc.annexes = self._extract_annexes()
        logger.info("Extracted %d annexes", len(doc.annexes))

        doc.all_links = self._collect_all_links(doc)
        logger.info("Collected %d total cross-references", len(doc.all_links))

        return doc

    # ------------------------------------------------------------------
    # Title & CELEX
    # ------------------------------------------------------------------

    def _extract_title(self) -> str:
        """Extract the document title using the schema selector."""
        title_elem = self.soup.select_one(self.schema.title_selector)
        if title_elem:
            return title_elem.get_text(strip=True)
        return "Unknown Title"

    def _extract_celex(self) -> str:
        """Extract CELEX number from HTML meta tags or canonical link."""
        meta = self.soup.find("meta", {"name": "celex"})
        if meta and meta.get("content"):
            return meta["content"]

        for link in self.soup.find_all("link", {"rel": "canonical"}):
            href = link.get("href", "")
            match = re.search(r"CELEX[:%](\d{5}[A-Z]\d{4})", href)
            if match:
                return match.group(1)

        return "unknown"

    # ------------------------------------------------------------------
    # Recitals
    # ------------------------------------------------------------------

    def _extract_recitals(self) -> list[Recital]:
        """Extract recitals -- override to change recital parsing logic."""
        recitals: list[Recital] = []

        for container in self.soup.select(self.schema.recital_container_selector):
            recital_id = container.get("id", "")
            number = re.search(self.schema.recital_id_pattern, recital_id)
            num_str = number.group(1) if number else str(len(recitals) + 1)

            text_parts: list[str] = []
            links: list[CrossReference] = []

            for p in container.select("p"):
                text_parts.append(p.get_text(strip=True))
                links.extend(self._extract_links(p, "recital", f"rec_{num_str}"))

            if text_parts:
                recitals.append(
                    Recital(
                        number=num_str,
                        text=" ".join(text_parts),
                        links=links,
                    )
                )

        # Fallback when no <div> containers are found
        if not recitals:
            recitals = self._extract_recitals_fallback()

        return recitals

    def _extract_recitals_fallback(self) -> list[Recital]:
        """
        Fallback recital extraction using ``(N)`` paragraph numbering.

        Override to handle documents with non-standard preamble layout.
        """
        recitals: list[Recital] = []
        recital_pattern = re.compile(r"^\((\d+)\)\s*")

        for p in self.soup.select(self.schema.recital_text_selector):
            text = p.get_text(strip=True)
            match = recital_pattern.match(text)
            if match:
                num = match.group(1)
                content = recital_pattern.sub("", text)
                links = self._extract_links(p, "recital", f"rec_{num}")
                recitals.append(Recital(number=num, text=content, links=links))

        return recitals

    # ------------------------------------------------------------------
    # Body (chapters / sections / articles)
    # ------------------------------------------------------------------

    def _extract_body(self) -> tuple[list[Chapter], list[Article]]:
        """Extract the main body structure."""
        chapters: list[Chapter] = []
        standalone_articles: list[Article] = []

        current_chapter: Chapter | None = None
        current_section: Section | None = None

        structural_elements = self.soup.select(self.schema.structural_selector)

        for elem in structural_elements:
            text = elem.get_text(strip=True) if isinstance(elem, Tag) else ""

            # --- Chapter ---
            if self._is_chapter_heading(elem, text):
                current_chapter = self._parse_chapter_heading(elem, text, len(chapters))
                chapters.append(current_chapter)
                current_section = None
                self._current_chapter = current_chapter.number
                logger.debug("Found chapter: %s", current_chapter.number)

            # --- Section ---
            elif self._is_section_heading(elem):
                current_section = self._parse_section_heading(
                    elem,
                    text,
                    len(current_chapter.sections if current_chapter else []),
                )
                if current_chapter:
                    current_chapter.sections.append(current_section)
                self._current_section = current_section.number
                logger.debug("Found section: %s", current_section.number)

            # --- Article ---
            elif self._is_article_element(elem):
                article = self._parse_article(elem)
                if article:
                    article.chapter = self._current_chapter
                    article.section = self._current_section

                    if current_section:
                        current_section.articles.append(article)
                    elif current_chapter:
                        current_chapter.articles.append(article)
                    else:
                        standalone_articles.append(article)

                    logger.debug("Found article: %s", article.number)

        return chapters, standalone_articles

    # Heading detection helpers (override for alternative structures) --------

    def _is_chapter_heading(self, elem: Tag, text: str) -> bool:
        cls_token = self.schema.chapter_selector.split(",")[0].strip().lstrip("p.")
        return (
            cls_token in elem.get("class", [])
            and self.schema.chapter_keyword.upper() in text.upper()
        )

    def _is_section_heading(self, elem: Tag) -> bool:
        cls_token = self.schema.section_selector.split(",")[0].strip().lstrip("p.")
        return cls_token in elem.get("class", [])

    def _is_article_element(self, elem: Tag) -> bool:
        cls_token = self.schema.article_title_selector.split(",")[0].strip().lstrip("p.")
        return cls_token in elem.get("class", []) or elem.get("id", "").startswith("art")

    # Heading parsers -------------------------------------------------------

    def _parse_chapter_heading(self, elem: Tag, text: str, index: int) -> Chapter:
        chapter_match = re.search(self.schema.chapter_number_pattern, text, re.I)
        title_match = re.search(self.schema.chapter_title_pattern, text, re.I)
        return Chapter(
            number=(chapter_match.group(1) if chapter_match else str(index + 1)),
            title=title_match.group(1).strip() if title_match else text,
        )

    def _parse_section_heading(self, elem: Tag, text: str, index: int) -> Section:
        section_match = re.search(self.schema.section_number_pattern, text, re.I)
        return Section(
            number=(section_match.group(1) if section_match else str(index + 1)),
            title=text,
        )

    # ------------------------------------------------------------------
    # Article parsing
    # ------------------------------------------------------------------

    def _parse_article(self, elem: Tag) -> Article | None:
        """Parse a single article element."""
        text = elem.get_text(strip=True)
        article_match = re.search(self.schema.article_number_pattern, text, re.I)
        if not article_match:
            article_id = elem.get("id", "")
            id_match = re.search(self.schema.article_id_pattern, article_id, re.I)
            if id_match:
                article_num = id_match.group(1)
            else:
                return None
        else:
            article_num = article_match.group(1)

        # Subtitle / title
        # EUR-Lex structures can have:
        # 1. Direct sibling: <p class="oj-ti-art"> followed by <p class="oj-sti-art">
        # 2. Nested in div: <p class="oj-ti-art"> followed by <div class="eli-title"><p class="oj-sti-art">
        title = ""
        subtitle_cls = self.schema.article_subtitle_selector.lstrip("p.")

        # Try direct sibling first
        subtitle_elem = elem.find_next_sibling("p", class_=subtitle_cls)

        # If not found, check inside next sibling div.eli-title
        if not subtitle_elem:
            title_div = elem.find_next_sibling("div", class_="eli-title")
            if title_div:
                subtitle_elem = title_div.find("p", class_=subtitle_cls)

        if subtitle_elem:
            title = subtitle_elem.get_text(strip=True)

        article = Article(number=article_num, title=title)
        article.paragraphs = self._extract_article_paragraphs(elem, article_num)
        return article

    def _extract_article_paragraphs(self, article_elem: Tag, article_num: str) -> list[Paragraph]:
        """Extract paragraphs from an article element.

        EUR-Lex uses two structures for sub-points:
        1. Inline: ``(a) text content`` in a single <p>
        2. Table-based: <table><tr><td>(a)</td><td>text content</td></tr></table>

        This method handles both by:
        - Selecting all paragraph elements
        - Skipping standalone point markers (e.g., just "(a)")
        - Processing tables to combine marker + content cells
        """
        paragraphs: list[Paragraph] = []
        source_id = f"art_{article_num}"

        container = article_elem.find_parent("div", class_="eli-subdivision")
        if not container:
            return paragraphs

        para_pattern = re.compile(r"^(\d+)\.\s*")
        # Matches standalone markers: (a), (1), (ii), (iii), etc.
        point_standalone_pattern = re.compile(r"^\(([a-z]+|\d+|[ivx]+)\)$")
        # Matches markers with optional content following
        point_inline_pattern = re.compile(r"^\(([a-z]+|\d+|[ivx]+)\)\s*")

        current_para_num = 0

        # First pass: get all paragraphs (excluding standalone markers which are in tables)
        for p_elem in container.select(self.schema.paragraph_selector):
            # Skip article title elements
            if "oj-ti-art" in p_elem.get("class", []):
                continue
            if "oj-sti-art" in p_elem.get("class", []):
                continue

            text = p_elem.get_text(strip=True)
            if not text:
                continue

            # Skip standalone point markers — they're handled via table processing
            if point_standalone_pattern.match(text):
                continue

            para_match = para_pattern.match(text)
            inline_point_match = point_inline_pattern.match(text)

            if para_match:
                current_para_num = int(para_match.group(1))
                content = para_pattern.sub("", text)
                num_str = str(current_para_num)
            elif inline_point_match:
                # Inline point with content
                content = point_inline_pattern.sub("", text)
                num_str = f"{current_para_num}({inline_point_match.group(1)})"
            else:
                content = text
                num_str = str(current_para_num) if current_para_num else "intro"

            # Check if this paragraph is inside a table (content cell)
            # If so, skip it here — it will be handled in table processing
            parent_td = p_elem.find_parent("td")
            if parent_td:
                # Check if there's a sibling td with a point marker
                parent_tr = parent_td.find_parent("tr")
                if parent_tr:
                    cells = parent_tr.find_all("td")
                    if len(cells) >= 2:
                        first_cell_text = cells[0].get_text(strip=True)
                        if point_standalone_pattern.match(first_cell_text):
                            continue  # Skip, will be handled in table pass

            links = self._extract_links(p_elem, "article", source_id)
            paragraphs.append(Paragraph(number=num_str, text=content, links=links))

        # Second pass: process tables for point lists
        # Only process top-level tables (not nested inside other tables)
        # to avoid duplicates from nested table structures
        all_tables = container.find_all("table")
        nested_table_ids = set()
        for table in all_tables:
            for inner_table in table.find_all("table"):
                nested_table_ids.add(id(inner_table))

        top_level_tables = [t for t in all_tables if id(t) not in nested_table_ids]

        # Pattern for numeric point markers like (1), (2), (45)
        numeric_point_pattern = re.compile(r"^\((\d+)\)$")
        # Pattern for letter point markers like (a), (b), (i), (ii)
        letter_point_pattern = re.compile(r"^\([a-z]+\)$|^\([ivx]+\)$")

        current_para_num = 0
        for table in top_level_tables:
            rows = table.find_all("tr")
            if not rows:
                continue

            # Check first row to determine table type
            first_cells = rows[0].find_all("td")
            if len(first_cells) < 2:
                continue

            first_marker = first_cells[0].get_text(strip=True)

            # Determine current_para_num for this table by looking at preceding paragraph
            prev_para = table.find_previous_sibling(
                lambda t: isinstance(t, Tag) and t.name in ("p", "div")
            )
            if prev_para:
                prev_text = prev_para.get_text(strip=True)
                prev_match = para_pattern.match(prev_text)
                if prev_match:
                    current_para_num = int(prev_match.group(1))

            # Skip letter-marked tables at the top level ONLY when numbered
            # paragraphs have already been collected (meaning these tables are
            # duplicates of inline content).  When no numbered paragraphs exist
            # (e.g. DSA Article 3 Definitions), the letter tables ARE the
            # primary content and must be processed.
            if letter_point_pattern.match(first_marker) and current_para_num == 0:
                has_numbered = any(re.match(r"^\d+$", p.number) for p in paragraphs)
                if has_numbered:
                    continue

            # Check if first row has a numeric marker with nested content
            # (e.g., definition with embedded sub-points)
            first_row_has_nested_def = False
            if numeric_point_pattern.match(first_marker):
                # Check if content cell has nested tables (sub-points)
                has_nested = len(first_cells[1].find_all("table")) > 0
                first_row_has_nested_def = has_nested

            for row_idx, row in enumerate(rows):
                # If first row has nested definition, skip subsequent rows
                # (sub-points are already captured in first row's get_text())
                if first_row_has_nested_def and row_idx > 0:
                    continue

                cells = row.find_all("td")
                if len(cells) >= 2:
                    marker_text = cells[0].get_text(strip=True)
                    content_text = cells[1].get_text(strip=True)

                    marker_match = point_standalone_pattern.match(marker_text)
                    if marker_match and content_text:
                        num_str = f"{current_para_num}({marker_match.group(1)})"
                        links = self._extract_links(cells[1], "article", source_id)
                        paragraphs.append(Paragraph(number=num_str, text=content_text, links=links))

        # Sort paragraphs by number to maintain proper order
        def sort_key(p):
            # Parse "4(a)" → (4, 0, 'a'), "4(10)" → (4, 10, ''), "4" → (4, -1, ''), "intro" → (-1, -1, '')
            # Using -1 for missing parts to sort before any actual values
            m = re.match(r"(\d+)(?:\(([a-z]+|\d+)\))?", p.number)
            if m:
                main_num = int(m.group(1))
                sub = m.group(2) or ""
                if sub.isdigit():
                    # Numeric sub-point: (1), (10), etc. - sort numerically
                    return (main_num, int(sub), "")
                else:
                    # Letter sub-point: (a), (b), etc. - sort alphabetically
                    return (main_num, 0, sub)
            return (-1, -1, p.number)

        paragraphs.sort(key=sort_key)

        return paragraphs

    # ------------------------------------------------------------------
    # Annexes
    # ------------------------------------------------------------------

    def _extract_annexes(self) -> list[Annex]:
        """Extract annexes from the document."""
        annexes: list[Annex] = []

        for container in self.soup.select(self.schema.annex_container_selector):
            annex_id = container.get("id", "")
            number_match = re.search(self.schema.annex_id_pattern, annex_id, re.I)
            number = number_match.group(1) if number_match else str(len(annexes) + 1)

            title_elem = container.select_one(self.schema.annex_title_selector)
            title = title_elem.get_text(strip=True) if title_elem else f"Annex {number}"

            content_parts: list[str] = []
            links: list[CrossReference] = []
            for p in container.select("p"):
                content_parts.append(p.get_text(strip=True))
                links.extend(self._extract_links(p, "annex", f"anx_{number}"))

            annexes.append(
                Annex(
                    number=number,
                    title=title,
                    content="\n".join(content_parts),
                    links=links,
                )
            )

        return annexes

    # ------------------------------------------------------------------
    # Link extraction (shared)
    # ------------------------------------------------------------------

    def _extract_links(
        self, element: Tag, source_type: str, source_id: str
    ) -> list[CrossReference]:
        """Extract all hyperlinks from an HTML element."""
        links: list[CrossReference] = []

        for a_tag in element.find_all("a", href=True):
            href = a_tag["href"]
            anchor_text = a_tag.get_text(strip=True)

            if href.startswith("/") or href.startswith("./"):
                href = urljoin(self.base_url, href)

            link_type, target_id = classify_link(href, self.base_url)

            links.append(
                CrossReference(
                    source_type=source_type,
                    source_id=source_id,
                    target_url=href,
                    anchor_text=anchor_text,
                    link_type=link_type,
                    target_id=target_id,
                )
            )

        return links

    # ------------------------------------------------------------------
    # Link aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_all_links(doc: Document) -> list[CrossReference]:
        """Flatten all cross-references into a single list."""
        all_links: list[CrossReference] = []

        for recital in doc.recitals:
            all_links.extend(recital.links)

        for chapter in doc.chapters:
            for article in chapter.articles:
                for para in article.paragraphs:
                    all_links.extend(para.links)
            for section in chapter.sections:
                for article in section.articles:
                    for para in article.paragraphs:
                        all_links.extend(para.links)

        for article in doc.articles:
            for para in article.paragraphs:
                all_links.extend(para.links)

        for annex in doc.annexes:
            all_links.extend(annex.links)

        return all_links
