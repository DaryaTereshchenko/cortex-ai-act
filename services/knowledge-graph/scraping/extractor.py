"""
EUR-Lex document structure extractor.

Parses HTML from EUR-Lex to extract recitals, articles, and cross-references
in a structured format suitable for knowledge graph ingestion.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup, Tag

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class LinkType(Enum):
    """Classification of EUR-Lex hyperlinks."""

    INTERNAL_ARTICLE = "internal_article"
    INTERNAL_RECITAL = "internal_recital"
    INTERNAL_ANNEX = "internal_annex"
    INTERNAL_OTHER = "internal_other"
    EXTERNAL_EU_LAW = "external_eu_law"
    EXTERNAL_ELI = "external_eli"
    EXTERNAL_OTHER = "external_other"


@dataclass
class CrossReference:
    """Represents a hyperlink/cross-reference in the document."""

    source_type: str  # "article", "recital", "annex", "preamble"
    source_id: str  # e.g., "art_6", "rec_47"
    target_url: str
    anchor_text: str
    link_type: LinkType
    target_id: str | None = None  # Parsed target identifier if internal

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "target_url": self.target_url,
            "anchor_text": self.anchor_text,
            "link_type": self.link_type.value,
            "target_id": self.target_id,
        }


@dataclass
class Paragraph:
    """A single paragraph or point within an article."""

    number: str  # e.g., "1", "2(a)", "2(b)(i)"
    text: str
    links: list[CrossReference] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "text": self.text,
            "links": [link.to_dict() for link in self.links],
        }


@dataclass
class Article:
    """An article in the regulation."""

    number: str  # e.g., "6", "52"
    title: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    chapter: str | None = None
    section: str | None = None

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "chapter": self.chapter,
            "section": self.section,
            "paragraphs": [p.to_dict() for p in self.paragraphs],
        }


@dataclass
class Recital:
    """A recital from the preamble."""

    number: str  # e.g., "1", "47"
    text: str
    links: list[CrossReference] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "text": self.text,
            "links": [link.to_dict() for link in self.links],
        }


@dataclass
class Chapter:
    """A chapter in the regulation."""

    number: str  # e.g., "I", "II", "III"
    title: str
    sections: list[Section] = field(default_factory=list)
    articles: list[Article] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "articles": [a.to_dict() for a in self.articles],
        }


@dataclass
class Section:
    """A section within a chapter."""

    number: str
    title: str
    articles: list[Article] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "articles": [a.to_dict() for a in self.articles],
        }


@dataclass
class Annex:
    """An annex to the regulation."""

    number: str  # e.g., "I", "III"
    title: str
    content: str
    links: list[CrossReference] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "links": [link.to_dict() for link in self.links],
        }


@dataclass
class Document:
    """Complete parsed EUR-Lex document."""

    title: str
    celex_number: str  # e.g., "32024R1689"
    recitals: list[Recital] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    articles: list[Article] = field(default_factory=list)  # Standalone articles
    annexes: list[Annex] = field(default_factory=list)
    all_links: list[CrossReference] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "celex_number": self.celex_number,
            "recitals": [r.to_dict() for r in self.recitals],
            "chapters": [c.to_dict() for c in self.chapters],
            "articles": [a.to_dict() for a in self.articles],
            "annexes": [a.to_dict() for a in self.annexes],
            "all_links": [link.to_dict() for link in self.all_links],
            "stats": {
                "recital_count": len(self.recitals),
                "chapter_count": len(self.chapters),
                "article_count": self._count_articles(),
                "annex_count": len(self.annexes),
                "total_links": len(self.all_links),
            },
        }

    def _count_articles(self) -> int:
        count = len(self.articles)
        for chapter in self.chapters:
            count += len(chapter.articles)
            for section in chapter.sections:
                count += len(section.articles)
        return count


# ---------------------------------------------------------------------------
# Link Classification
# ---------------------------------------------------------------------------

# Patterns for identifying link types
INTERNAL_PATTERNS = {
    r"#art[_-]?(\d+)": LinkType.INTERNAL_ARTICLE,
    r"#rec[_-]?(\d+)": LinkType.INTERNAL_RECITAL,
    r"#anx[_-]?([IVX]+|\d+)": LinkType.INTERNAL_ANNEX,
    r"^#": LinkType.INTERNAL_OTHER,
}

EXTERNAL_EU_PATTERN = re.compile(
    r"eur-lex\.europa\.eu.*CELEX[:%](\d{5}[A-Z]\d{4})"
)
ELI_PATTERN = re.compile(r"data\.europa\.eu/eli/")


def classify_link(href: str, base_url: str = "") -> tuple[LinkType, str | None]:
    """
    Classify a hyperlink and extract target ID if applicable.

    Args:
        href: The href attribute value.
        base_url: Base URL for resolving relative links.

    Returns:
        Tuple of (LinkType, target_id or None).
    """
    if not href:
        return LinkType.EXTERNAL_OTHER, None

    # Internal links (same document)
    for pattern, link_type in INTERNAL_PATTERNS.items():
        match = re.search(pattern, href, re.IGNORECASE)
        if match:
            target_id = match.group(1) if match.lastindex else href.lstrip("#")
            return link_type, target_id

    # External EU legislation
    eu_match = EXTERNAL_EU_PATTERN.search(href)
    if eu_match:
        return LinkType.EXTERNAL_EU_LAW, eu_match.group(1)

    # ELI URIs
    if ELI_PATTERN.search(href):
        return LinkType.EXTERNAL_ELI, None

    # Other external
    return LinkType.EXTERNAL_OTHER, None


# ---------------------------------------------------------------------------
# EUR-Lex HTML Parser
# ---------------------------------------------------------------------------


class EURLexExtractor:
    """
    Extracts structured data from EUR-Lex HTML documents.

    EUR-Lex CSS class patterns:
    - `.ti-grseq-1`: Chapter titles (CHAPTER I, II, etc.)
    - `.ti-section-1`: Section titles
    - `.ti-art`: Article titles ("Article 6")
    - `.sti-art`: Article subtitles (article name)
    - `.normal`: Regular paragraph text
    - `.li`: List items (points/subpoints)
    """

    # CSS selectors for EUR-Lex structure
    SELECTORS = {
        "title": "p.title-doc-first, p.doc-ti",
        "recital_container": "div.eli-subdivision[id^='rct']",
        "recital_text": "p.normal",
        "chapter": "p.ti-grseq-1",
        "section": "p.ti-section-1",
        "article_title": "p.ti-art",
        "article_subtitle": "p.sti-art",
        "paragraph": "p.normal, p.li",
        "annex_title": "p.ti-grseq-1[id^='anx']",
    }

    def __init__(self, soup: BeautifulSoup, base_url: str = ""):
        """
        Initialize extractor with parsed HTML.

        Args:
            soup: BeautifulSoup parsed document.
            base_url: Base URL for resolving relative links.
        """
        self.soup = soup
        self.base_url = base_url
        self._current_chapter: str | None = None
        self._current_section: str | None = None

    def extract(self) -> Document:
        """
        Extract all structured data from the document.

        Returns:
            Complete Document object with all parsed elements.
        """
        logger.info("Starting document extraction")

        doc = Document(
            title=self._extract_title(),
            celex_number=self._extract_celex(),
        )

        # Extract recitals
        doc.recitals = self._extract_recitals()
        logger.info("Extracted %d recitals", len(doc.recitals))

        # Extract main body (chapters/sections/articles)
        doc.chapters, doc.articles = self._extract_body()
        logger.info(
            "Extracted %d chapters, %d standalone articles",
            len(doc.chapters),
            len(doc.articles),
        )

        # Extract annexes
        doc.annexes = self._extract_annexes()
        logger.info("Extracted %d annexes", len(doc.annexes))

        # Collect all links
        doc.all_links = self._collect_all_links(doc)
        logger.info("Collected %d total cross-references", len(doc.all_links))

        return doc

    def _extract_title(self) -> str:
        """Extract document title."""
        title_elem = self.soup.select_one(self.SELECTORS["title"])
        if title_elem:
            return title_elem.get_text(strip=True)
        return "Unknown Title"

    def _extract_celex(self) -> str:
        """Extract CELEX number from URL or metadata."""
        # Try meta tag
        meta = self.soup.find("meta", {"name": "celex"})
        if meta and meta.get("content"):
            return meta["content"]

        # Try to find in URL-like patterns
        for link in self.soup.find_all("link", {"rel": "canonical"}):
            href = link.get("href", "")
            match = re.search(r"CELEX[:%](\d{5}[A-Z]\d{4})", href)
            if match:
                return match.group(1)

        return "unknown"

    def _extract_recitals(self) -> list[Recital]:
        """Extract all recitals from the preamble."""
        recitals = []

        # Method 1: Look for eli-subdivision containers
        for container in self.soup.select("div.eli-subdivision[id^='rct']"):
            recital_id = container.get("id", "")
            number = re.search(r"rct[_-]?(\d+)", recital_id)
            num_str = number.group(1) if number else str(len(recitals) + 1)

            text_parts = []
            links = []

            for p in container.select("p"):
                text_parts.append(p.get_text(strip=True))
                links.extend(self._extract_links(p, "recital", f"rec_{num_str}"))

            if text_parts:
                recitals.append(Recital(
                    number=num_str,
                    text=" ".join(text_parts),
                    links=links,
                ))

        # Method 2: Fallback - look for numbered paragraphs in preamble
        if not recitals:
            recitals = self._extract_recitals_fallback()

        return recitals

    def _extract_recitals_fallback(self) -> list[Recital]:
        """Fallback recital extraction using paragraph numbering."""
        recitals = []
        recital_pattern = re.compile(r"^\((\d+)\)\s*")

        for p in self.soup.select("p.normal"):
            text = p.get_text(strip=True)
            match = recital_pattern.match(text)
            if match:
                num = match.group(1)
                content = recital_pattern.sub("", text)
                links = self._extract_links(p, "recital", f"rec_{num}")
                recitals.append(Recital(number=num, text=content, links=links))

        return recitals

    def _extract_body(self) -> tuple[list[Chapter], list[Article]]:
        """
        Extract the main body structure (chapters, sections, articles).

        Returns:
            Tuple of (chapters list, standalone articles list).
        """
        chapters: list[Chapter] = []
        standalone_articles: list[Article] = []

        current_chapter: Chapter | None = None
        current_section: Section | None = None

        # Find all structural elements
        structural_elements = self.soup.select(
            "p.ti-grseq-1, p.ti-section-1, p.ti-art, div.eli-subdivision[id^='art']"
        )

        for elem in structural_elements:
            text = elem.get_text(strip=True) if isinstance(elem, Tag) else ""

            # Chapter detection
            if "ti-grseq-1" in elem.get("class", []) and "CHAPTER" in text.upper():
                chapter_match = re.search(r"CHAPTER\s+([IVXLCDM]+|\d+)", text, re.I)
                title_match = re.search(r"CHAPTER\s+[IVXLCDM\d]+\s*(.+)", text, re.I)

                current_chapter = Chapter(
                    number=chapter_match.group(1) if chapter_match else str(len(chapters) + 1),
                    title=title_match.group(1).strip() if title_match else text,
                )
                chapters.append(current_chapter)
                current_section = None
                self._current_chapter = current_chapter.number
                logger.debug("Found chapter: %s", current_chapter.number)

            # Section detection
            elif "ti-section-1" in elem.get("class", []):
                section_match = re.search(r"Section\s+(\d+)", text, re.I)
                current_section = Section(
                    number=section_match.group(1) if section_match else str(len(current_chapter.sections) + 1 if current_chapter else 1),
                    title=text,
                )
                if current_chapter:
                    current_chapter.sections.append(current_section)
                self._current_section = current_section.number
                logger.debug("Found section: %s", current_section.number)

            # Article detection
            elif "ti-art" in elem.get("class", []) or elem.get("id", "").startswith("art"):
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

    def _parse_article(self, elem: Tag) -> Article | None:
        """Parse a single article element."""
        # Get article number
        text = elem.get_text(strip=True)
        article_match = re.search(r"Article\s+(\d+)", text, re.I)
        if not article_match:
            # Try ID-based extraction
            article_id = elem.get("id", "")
            id_match = re.search(r"art[_-]?(\d+)", article_id, re.I)
            if id_match:
                article_num = id_match.group(1)
            else:
                return None
        else:
            article_num = article_match.group(1)

        # Find article title (subtitle element)
        title = ""
        subtitle_elem = elem.find_next_sibling("p", class_="sti-art")
        if subtitle_elem:
            title = subtitle_elem.get_text(strip=True)

        # Find article container or paragraphs
        article = Article(number=article_num, title=title)
        article.paragraphs = self._extract_article_paragraphs(elem, article_num)

        return article

    def _extract_article_paragraphs(self, article_elem: Tag, article_num: str) -> list[Paragraph]:
        """Extract paragraphs from an article."""
        paragraphs = []
        source_id = f"art_{article_num}"

        # Look for the containing div or subsequent paragraphs
        container = article_elem.find_parent("div", class_="eli-subdivision")
        if container:
            para_elems = container.select("p.normal, p.li, table.li")
        else:
            # Find siblings until next article
            para_elems = []
            for sibling in article_elem.find_next_siblings():
                if isinstance(sibling, Tag):
                    if "ti-art" in sibling.get("class", []):
                        break
                    if sibling.name == "p":
                        para_elems.append(sibling)

        para_pattern = re.compile(r"^(\d+)\.\s*")
        point_pattern = re.compile(r"^\(([a-z]|\d+)\)\s*")

        current_para_num = 0
        for p_elem in para_elems:
            text = p_elem.get_text(strip=True)
            if not text:
                continue

            # Check for numbered paragraph
            para_match = para_pattern.match(text)
            point_match = point_pattern.match(text)

            if para_match:
                current_para_num = int(para_match.group(1))
                content = para_pattern.sub("", text)
                num_str = str(current_para_num)
            elif point_match:
                content = point_pattern.sub("", text)
                num_str = f"{current_para_num}({point_match.group(1)})"
            else:
                content = text
                num_str = str(current_para_num) if current_para_num else "intro"

            links = self._extract_links(p_elem, "article", source_id)
            paragraphs.append(Paragraph(number=num_str, text=content, links=links))

        return paragraphs

    def _extract_annexes(self) -> list[Annex]:
        """Extract annexes from the document."""
        annexes = []

        for container in self.soup.select("div.eli-subdivision[id^='anx']"):
            annex_id = container.get("id", "")
            number_match = re.search(r"anx[_-]?([IVXLCDM]+|\d+)", annex_id, re.I)
            number = number_match.group(1) if number_match else str(len(annexes) + 1)

            # Get title
            title_elem = container.select_one("p.ti-grseq-1, p.title")
            title = title_elem.get_text(strip=True) if title_elem else f"Annex {number}"

            # Get content
            content_parts = []
            links = []
            for p in container.select("p"):
                content_parts.append(p.get_text(strip=True))
                links.extend(self._extract_links(p, "annex", f"anx_{number}"))

            annexes.append(Annex(
                number=number,
                title=title,
                content="\n".join(content_parts),
                links=links,
            ))

        return annexes

    def _extract_links(
        self, element: Tag, source_type: str, source_id: str
    ) -> list[CrossReference]:
        """
        Extract all hyperlinks from an element.

        Args:
            element: BeautifulSoup element to search.
            source_type: Type of source element (article, recital, etc.).
            source_id: ID of source element.

        Returns:
            List of CrossReference objects.
        """
        links = []

        for a_tag in element.find_all("a", href=True):
            href = a_tag["href"]
            anchor_text = a_tag.get_text(strip=True)

            # Resolve relative URLs
            if href.startswith("/") or href.startswith("./"):
                href = urljoin(self.base_url, href)

            link_type, target_id = classify_link(href, self.base_url)

            links.append(CrossReference(
                source_type=source_type,
                source_id=source_id,
                target_url=href,
                anchor_text=anchor_text,
                link_type=link_type,
                target_id=target_id,
            ))

        return links

    def _collect_all_links(self, doc: Document) -> list[CrossReference]:
        """Collect all links from the document into a flat list."""
        all_links = []

        # From recitals
        for recital in doc.recitals:
            all_links.extend(recital.links)

        # From articles (in chapters/sections)
        for chapter in doc.chapters:
            for article in chapter.articles:
                for para in article.paragraphs:
                    all_links.extend(para.links)
            for section in chapter.sections:
                for article in section.articles:
                    for para in article.paragraphs:
                        all_links.extend(para.links)

        # From standalone articles
        for article in doc.articles:
            for para in article.paragraphs:
                all_links.extend(para.links)

        # From annexes
        for annex in doc.annexes:
            all_links.extend(annex.links)

        return all_links


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def extract_from_html(
    html: str | BeautifulSoup,
    base_url: str = "",
) -> Document:
    """
    Extract structured data from EUR-Lex HTML.

    Args:
        html: Raw HTML string or parsed BeautifulSoup object.
        base_url: Base URL for resolving relative links.

    Returns:
        Parsed Document object.
    """
    if isinstance(html, str):
        soup = BeautifulSoup(html, "lxml")
    else:
        soup = html

    extractor = EURLexExtractor(soup, base_url)
    return extractor.extract()


def extract_from_url(url: str) -> Document:
    """
    Fetch and extract structured data from a EUR-Lex URL.

    Args:
        url: EUR-Lex document URL.

    Returns:
        Parsed Document object.
    """
    from .downloading import get_html_content

    soup = get_html_content(url)
    return extract_from_html(soup, base_url=url)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    # Add parent directory to path for logging_config import
    _service_root = Path(__file__).resolve().parent.parent
    if str(_service_root) not in sys.path:
        sys.path.insert(0, str(_service_root))

    from logging_config import configure_logging
    from downloading import get_ai_act_content, get_urls

    configure_logging(level=logging.DEBUG, log_file="extractor.log")
    logger.info("Starting EUR-Lex document extraction")

    try:
        # Fetch AI Act
        urls = get_urls()
        logger.info("Fetching EU AI Act from EUR-Lex...")
        soup = get_ai_act_content()

        # Extract structure
        logger.info("Extracting document structure...")
        doc = extract_from_html(soup, base_url=urls["AI_ACT_URL"])

        # Output summary
        print("\n" + "=" * 60)
        print(f"Document: {doc.title}")
        print(f"CELEX: {doc.celex_number}")
        print("=" * 60)
        print(f"Recitals: {len(doc.recitals)}")
        print(f"Chapters: {len(doc.chapters)}")
        print(f"Standalone Articles: {len(doc.articles)}")
        print(f"Annexes: {len(doc.annexes)}")
        print(f"Total Cross-References: {len(doc.all_links)}")
        print("=" * 60)

        # Link type breakdown
        link_counts = {}
        for link in doc.all_links:
            link_type = link.link_type.value
            link_counts[link_type] = link_counts.get(link_type, 0) + 1

        print("\nLink Type Breakdown:")
        for link_type, count in sorted(link_counts.items()):
            print(f"  {link_type}: {count}")

        # Save to JSON
        output_path = _service_root / "data" / "ai_act_extracted.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("Saved extraction to %s", output_path)
        print(f"\nSaved to: {output_path}")

    except Exception as e:
        logger.exception("Extraction failed: %s", e)
        sys.exit(1)
