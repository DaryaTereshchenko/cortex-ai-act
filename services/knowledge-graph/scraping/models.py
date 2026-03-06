"""
Data models for EUR-Lex document parsing.

Shared data structures and link classification used across all document parsers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Link Classification
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


# Patterns for identifying link types
INTERNAL_PATTERNS: dict[str, LinkType] = {
    r"#art[_-]?(\d+)": LinkType.INTERNAL_ARTICLE,
    r"#rec[_-]?(\d+)": LinkType.INTERNAL_RECITAL,
    r"#anx[_-]?([IVX]+|\d+)": LinkType.INTERNAL_ANNEX,
    r"^#": LinkType.INTERNAL_OTHER,
}

EXTERNAL_EU_PATTERN = re.compile(r"eur-lex\.europa\.eu.*CELEX[:%](\d{5}[A-Z]\d{4})")
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
# Data Models
# ---------------------------------------------------------------------------


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
    document_type: str = ""  # e.g., "ai_act", "dsa"
    recitals: list[Recital] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    articles: list[Article] = field(default_factory=list)  # Standalone articles
    annexes: list[Annex] = field(default_factory=list)
    all_links: list[CrossReference] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "celex_number": self.celex_number,
            "document_type": self.document_type,
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
