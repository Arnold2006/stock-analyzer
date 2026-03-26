"""Web-scraping helpers for financial portals.

Currently provides lightweight RSS + HTML fetching utilities.
All network I/O goes through this module so it can be easily mocked in tests.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; StockAnalyzer/1.0; +https://github.com/Arnold2006/stock-analyzer)"
    )
}
_TIMEOUT = 10  # seconds
_ALLOWED_SCHEMES = {"http", "https"}


def _validate_url(url: str) -> None:
    """Raise :exc:`ValueError` if *url* is not an acceptable http/https URL.

    Parameters
    ----------
    url:
        URL to validate.

    Raises
    ------
    ValueError
        If the scheme is not ``http`` or ``https``, or if the netloc is empty.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.netloc:
        raise ValueError(f"Disallowed or invalid URL: {url!r}")


def fetch_rss(url: str) -> list[dict[str, str]]:
    """Fetch and parse an RSS feed, returning a list of entry dicts.

    Parameters
    ----------
    url:
        URL of the RSS / Atom feed.

    Returns
    -------
    list[dict[str, str]]
        Each dict has keys ``title``, ``link``, ``summary``, ``published``.
    """
    try:
        _validate_url(url)
        feed = feedparser.parse(url, request_headers=_HEADERS)
        entries = []
        for entry in feed.entries:
            entries.append(
                {
                    "title": getattr(entry, "title", ""),
                    "link": getattr(entry, "link", ""),
                    "summary": getattr(entry, "summary", ""),
                    "published": getattr(entry, "published", ""),
                }
            )
        logger.debug("Fetched %d entries from %s", len(entries), url)
        return entries
    except Exception as exc:
        logger.error("Error fetching RSS %s: %s", url, exc)
        return []


def fetch_html(url: str) -> str:
    """Fetch an HTML page and return its raw text.

    Parameters
    ----------
    url:
        Target URL.

    Returns
    -------
    str
        Raw HTML string, or empty string on failure.
    """
    try:
        _validate_url(url)
        response = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        response.raise_for_status()
        logger.debug("Fetched HTML from %s (%d bytes)", url, len(response.text))
        return response.text
    except Exception as exc:
        logger.error("Error fetching HTML %s: %s", url, exc)
        return ""


def extract_text_from_html(html: str) -> str:
    """Strip HTML tags and return visible text.

    Parameters
    ----------
    html:
        Raw HTML string.

    Returns
    -------
    str
        Plain text extracted from the HTML.
    """
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator=" ", strip=True)


def fetch_yahoo_rss(ticker: str) -> list[dict[str, str]]:
    """Fetch the Yahoo Finance news RSS feed for *ticker*.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (e.g. ``"AAPL"``).

    Returns
    -------
    list[dict[str, str]]
        News entries from Yahoo Finance RSS.
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    return fetch_rss(url)


def fetch_marketwatch_rss() -> list[dict[str, str]]:
    """Fetch MarketWatch latest news RSS feed.

    Returns
    -------
    list[dict[str, str]]
        News entries from MarketWatch.
    """
    url = "https://feeds.marketwatch.com/marketwatch/topstories/"
    return fetch_rss(url)


def fetch_reuters_rss() -> list[dict[str, str]]:
    """Fetch Reuters business news RSS feed.

    Returns
    -------
    list[dict[str, str]]
        News entries from Reuters.
    """
    url = "https://feeds.reuters.com/reuters/businessNews"
    return fetch_rss(url)


def filter_entries_by_ticker(
    entries: list[dict[str, str]], ticker: str
) -> list[dict[str, str]]:
    """Return only entries whose title or summary mentions *ticker*.

    Parameters
    ----------
    entries:
        List of news entry dicts.
    ticker:
        Ticker symbol to search for (case-insensitive).

    Returns
    -------
    list[dict[str, str]]
        Filtered list of matching entries.
    """
    ticker_upper = ticker.upper()
    return [
        e
        for e in entries
        if ticker_upper in e.get("title", "").upper()
        or ticker_upper in e.get("summary", "").upper()
    ]
