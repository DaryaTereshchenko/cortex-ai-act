"""
Scraping module for EU regulatory documents.

Provides utilities for fetching and parsing HTML content from EUR-Lex,
with pluggable parsers for different regulations (AI Act, DSA, etc.).
"""

from .downloading import (
    ScrapingError,
    get_ai_act_content,
    get_dsa_content,
    get_html_content,
    get_urls,
)
from .extractor import BaseDocumentExtractor
from .models import (
    Annex,
    Article,
    Chapter,
    CrossReference,
    Document,
    LinkType,
    Paragraph,
    Recital,
    Section,
    classify_link,
)
from .parsers import (
    AIActExtractor,
    DSAExtractor,
    EURLexExtractor,
    create_extractor,
    extract_from_html,
    extract_from_url,
    register_parser,
)
from .schemas import (
    AI_ACT_SCHEMA,
    DSA_SCHEMA,
    SCHEMA_REGISTRY,
    ParsingSchema,
    get_schema,
)

__all__ = [
    "AI_ACT_SCHEMA",
    "AIActExtractor",
    "Annex",
    "Article",
    "BaseDocumentExtractor",
    "Chapter",
    "CrossReference",
    "DSA_SCHEMA",
    "DSAExtractor",
    "Document",
    "EURLexExtractor",
    "LinkType",
    "Paragraph",
    "ParsingSchema",
    "Recital",
    "SCHEMA_REGISTRY",
    "ScrapingError",
    "Section",
    "classify_link",
    "create_extractor",
    "extract_from_html",
    "extract_from_url",
    "get_ai_act_content",
    "get_dsa_content",
    "get_html_content",
    "get_schema",
    "get_urls",
    "register_parser",
]

