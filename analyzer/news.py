"""News aggregation module.

Fetches and deduplicates financial news headlines for a given ticker from
multiple RSS sources (Yahoo Finance, Reuters, MarketWatch).
"""

from __future__ import annotations

import logging
from typing import Any

from utils.cache import cached
from utils.scraping import (
    fetch_marketwatch_rss,
    fetch_reuters_rss,
    fetch_yahoo_rss,
    filter_entries_by_ticker,
)

logger = logging.getLogger(__name__)

MAX_HEADLINES = 10


@cached(ttl=300)
def get_news(ticker: str) -> list[dict[str, str]]:
    """Aggregate news about *ticker* from several RSS sources.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (e.g. ``"AAPL"``).

    Returns
    -------
    list[dict[str, str]]
        Deduplicated list of news entry dicts (keys: ``title``, ``link``,
        ``summary``, ``published``), newest first, capped at
        :data:`MAX_HEADLINES`.
    """
    ticker = ticker.strip().upper()
    all_entries: list[dict[str, str]] = []

    # Ticker-specific feed (Yahoo Finance)
    try:
        yahoo_entries = fetch_yahoo_rss(ticker)
        all_entries.extend(yahoo_entries)
        logger.info("Yahoo Finance returned %d entries for %s", len(yahoo_entries), ticker)
    except Exception as exc:
        logger.warning("Yahoo Finance RSS failed for %s: %s", ticker, exc)

    # General feeds – filtered by ticker mention
    for feed_func, name in [
        (fetch_reuters_rss, "Reuters"),
        (fetch_marketwatch_rss, "MarketWatch"),
    ]:
        try:
            entries = feed_func()
            filtered = filter_entries_by_ticker(entries, ticker)
            all_entries.extend(filtered)
            logger.info("%s returned %d relevant entries for %s", name, len(filtered), ticker)
        except Exception as exc:
            logger.warning("%s RSS failed for %s: %s", name, ticker, exc)

    # Deduplicate by title
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for entry in all_entries:
        title = entry.get("title", "").strip()
        if title and title not in seen:
            seen.add(title)
            unique.append(entry)

    result = unique[:MAX_HEADLINES]
    logger.info("Total unique headlines for %s: %d", ticker, len(result))
    return result
