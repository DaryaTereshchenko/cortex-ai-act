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
from .schemas import (
    AI_ACT_SCHEMA,
    DSA_SCHEMA,
    ParsingSchema,
    SCHEMA_REGISTRY,
    get_schema,
)
from .extractor import BaseDocumentExtractor
from .parsers import (
    AIActExtractor,
    DSAExtractor,
    EURLexExtractor,  # backward-compat alias
    create_extractor,
    extract_from_html,
    extract_from_url,
    register_parser,
)

__all__ = [
    # Downloading
    "ScrapingError",
    "get_ai_act_content",
    "get_dsa_content",
    "get_html_content",
    "get_urls",
    # Data models
    "Annex",
    "Article",
    "Chapter",
    "CrossReference",
    "Document",
    "LinkType",
    "Paragraph",
    "Recital",
    "Section",
    "classify_link",
    # Schemas
    "AI_ACT_SCHEMA",
    "DSA_SCHEMA",
    "ParsingSchema",
    "SCHEMA_REGISTRY",
    "get_schema",
    # Extractors
    "BaseDocumentExtractor",
    "AIActExtractor",
    "DSAExtractor",
    "EURLexExtractor",
    "create_extractor",
    "extract_from_html",
    "extract_from_url",
    "register_parser",
]
