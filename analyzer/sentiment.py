"""Sentiment analysis module.

Scores a list of news headlines using NLTK VADER, returning a compound
sentiment score in [-1, 1] and a label (bullish / neutral / bearish).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-load VADER to avoid import errors if nltk data is missing
_vader: Any = None


def _get_vader() -> Any:
    """Return a lazily-initialised VADER SentimentIntensityAnalyzer.

    Downloads the ``vader_lexicon`` corpus on first use if not present.

    Returns
    -------
    nltk.sentiment.vader.SentimentIntensityAnalyzer
        Initialised analyser instance.
    """
    global _vader
    if _vader is None:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            _vader = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            _vader = SentimentIntensityAnalyzer()
        logger.info("VADER SentimentIntensityAnalyzer initialised")
    return _vader


def score_text(text: str) -> float:
    """Return the VADER compound sentiment score for *text*.

    Parameters
    ----------
    text:
        The text to score.

    Returns
    -------
    float
        Compound score in the range [-1.0, 1.0].
    """
    try:
        sia = _get_vader()
        scores = sia.polarity_scores(text)
        return float(scores["compound"])
    except Exception as exc:
        logger.error("Sentiment scoring failed: %s", exc)
        return 0.0


def score_headlines(headlines: list[dict[str, str]]) -> dict[str, Any]:
    """Aggregate sentiment across multiple news headlines.

    Parameters
    ----------
    headlines:
        List of headline dicts with at least a ``"title"`` key.

    Returns
    -------
    dict[str, Any]
        - ``"compound"`` – average compound score [-1, 1]
        - ``"label"`` – ``"bullish"``, ``"neutral"``, or ``"bearish"``
        - ``"scores"`` – per-headline scores list
    """
    if not headlines:
        return {"compound": 0.0, "label": "neutral", "scores": []}

    scores = [score_text(h.get("title", "") + " " + h.get("summary", "")) for h in headlines]
    compound = sum(scores) / len(scores)

    if compound >= 0.05:
        label = "bullish"
    elif compound <= -0.05:
        label = "bearish"
    else:
        label = "neutral"

    logger.info(
        "Sentiment: compound=%.3f label=%s (from %d headlines)", compound, label, len(headlines)
    )
    return {"compound": compound, "label": label, "scores": scores}
