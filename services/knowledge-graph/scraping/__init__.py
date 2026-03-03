"""
Scraping module for EU regulatory documents.

Provides utilities for fetching and parsing HTML content from EUR-Lex.
"""

from .downloading import (
    ScrapingError,
    get_ai_act_content,
    get_dsa_content,
    get_html_content,
    get_urls,
)
from .extractor import (
    Annex,
    Article,
    Chapter,
    CrossReference,
    Document,
    EURLexExtractor,
    LinkType,
    Paragraph,
    Recital,
    Section,
    classify_link,
    extract_from_html,
    extract_from_url,
)

__all__ = [
    # Parser
    "ScrapingError",
    "get_ai_act_content",
    "get_dsa_content",
    "get_html_content",
    "get_urls",
    # Extractor - Data models
    "Article",
    "Annex",
    "Chapter",
    "CrossReference",
    "Document",
    "LinkType",
    "Paragraph",
    "Recital",
    "Section",
    # Extractor - Functions
    "EURLexExtractor",
    "classify_link",
    "extract_from_html",
    "extract_from_url",
]
