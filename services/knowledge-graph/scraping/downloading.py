"""
Web scraping module for EU regulatory documents.

Provides utilities for fetching and parsing HTML content from EUR-Lex.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from requests import Response

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LINKS_FILE = Path(__file__).parent / "links.json"
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_PARSER = "lxml"


def _load_urls() -> dict[str, str]:
    """Load URLs from the links.json configuration file."""
    try:
        with _LINKS_FILE.open() as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Links configuration file not found: %s", _LINKS_FILE)
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in links file: %s", e)
        raise


# Lazy-load URLs
_urls: dict[str, str] | None = None


def get_urls() -> dict[str, str]:
    """Get cached URL configuration."""
    global _urls
    if _urls is None:
        _urls = _load_urls()
    return _urls


# ---------------------------------------------------------------------------
# HTTP Client
# ---------------------------------------------------------------------------
class ScrapingError(Exception):
    """Raised when scraping fails."""

    def __init__(self, url: str, message: str, status_code: int | None = None):
        self.url = url
        self.status_code = status_code
        super().__init__(f"Failed to scrape {url}: {message}")


def fetch_url(url: str, *, timeout: int = DEFAULT_TIMEOUT) -> Response:
    """
    Fetch content from a URL with error handling.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        The HTTP response object.

    Raises:
        ScrapingError: If the request fails or returns a non-2xx status.
    """
    logger.info("Fetching URL: %s", url)

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout as e:
        logger.error("Request timed out for URL: %s", url)
        raise ScrapingError(url, "Request timed out") from e
    except requests.exceptions.ConnectionError as e:
        logger.error("Connection error for URL %s: %s", url, e)
        raise ScrapingError(url, f"Connection error: {e}") from e
    except requests.exceptions.HTTPError as e:
        logger.error("HTTP error %d for URL: %s", e.response.status_code, url)
        raise ScrapingError(url, str(e), status_code=e.response.status_code) from e

    logger.debug("Successfully fetched %d bytes from %s", len(response.content), url)
    return response


def get_html_content(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    parser: str = DEFAULT_PARSER,
) -> BeautifulSoup:
    """
    Fetch and parse HTML content from a URL.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.
        parser: BeautifulSoup parser to use (default: lxml).

    Returns:
        Parsed BeautifulSoup object.

    Raises:
        ScrapingError: If fetching or parsing fails.
    """
    response = fetch_url(url, timeout=timeout)

    try:
        soup = BeautifulSoup(response.text, parser)
    except Exception as e:
        logger.error("Failed to parse HTML from %s: %s", url, e)
        raise ScrapingError(url, f"HTML parsing failed: {e}") from e

    logger.info("Parsed HTML document from %s (%d elements)", url, len(soup.find_all()))
    return soup


def get_ai_act_content(*, timeout: int = DEFAULT_TIMEOUT) -> BeautifulSoup:
    """Fetch and parse the EU AI Act document."""
    urls = get_urls()
    return get_html_content(urls["AI_ACT_URL"], timeout=timeout)


def get_dsa_content(*, timeout: int = DEFAULT_TIMEOUT) -> BeautifulSoup:
    """Fetch and parse the Digital Services Act document."""
    urls = get_urls()
    return get_html_content(urls["DSA_URL"], timeout=timeout)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Add parent directory to path for logging_config import
    import sys

    _service_root = Path(__file__).resolve().parent.parent
    if str(_service_root) not in sys.path:
        sys.path.insert(0, str(_service_root))

    from logging_config import configure_logging

    configure_logging(level=logging.DEBUG, log_file="scraping.log")
    logger.info("Starting EU regulatory document scraper")

    try:
        logger.info("Fetching EU AI Act document...")
        html_content = get_ai_act_content()
        logger.info("Successfully scraped AI Act document")
        print(html_content.prettify()[:2000])  # Print first 2000 chars
    except ScrapingError as e:
        logger.error("Scraping failed: %s", e)
        sys.exit(1)
