"""Prediction module.

Combines sentiment, momentum, and volatility signals into a single
confidence score and generates a human-readable explanation.

Score formula
-------------
    score = (sentiment_weight * sentiment_component)
            + (momentum_weight * momentum_component)
            + (volatility_weight * volatility_component)

All components are normalised to [0, 1] before weighting.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Ensemble weights
SENTIMENT_WEIGHT = 0.40
MOMENTUM_WEIGHT = 0.40
VOLATILITY_WEIGHT = 0.20


def _sentiment_component(compound: float) -> float:
    """Convert VADER compound score [-1, 1] to a [0, 1] component.

    Parameters
    ----------
    compound:
        VADER compound sentiment score.

    Returns
    -------
    float
        Component value in [0, 1].
    """
    return (compound + 1.0) / 2.0


def _momentum_component(
    rsi: float | None,
    price_change_pct: float | None,
    sma_10: float | None,
    sma_50: float | None,
) -> float:
    """Estimate a [0, 1] momentum component from technical indicators.

    Parameters
    ----------
    rsi:
        14-day RSI (0–100), or ``None``.
    price_change_pct:
        5-day price change in percent, or ``None``.
    sma_10:
        10-day SMA, or ``None``.
    sma_50:
        50-day SMA, or ``None``.

    Returns
    -------
    float
        Momentum component in [0, 1].
    """
    sub_scores: list[float] = []

    # RSI: neutral at 50, bullish above, bearish below
    if rsi is not None:
        sub_scores.append(rsi / 100.0)

    # Price change over last 5 days
    if price_change_pct is not None:
        # clip to ±10 % then normalise to [0, 1]
        clipped = max(-10.0, min(10.0, price_change_pct))
        sub_scores.append((clipped + 10.0) / 20.0)

    # SMA cross
    if sma_10 is not None and sma_50 is not None:
        sub_scores.append(1.0 if sma_10 > sma_50 else 0.0)

    if not sub_scores:
        return 0.5  # neutral default
    return sum(sub_scores) / len(sub_scores)


def _volatility_component(volatility: float | None) -> float:
    """Convert annualised volatility to a [0, 1] component.

    Higher volatility is penalised (more risk → lower score contribution).

    Parameters
    ----------
    volatility:
        Annualised volatility (fraction), or ``None``.

    Returns
    -------
    float
        Component value in [0, 1].
    """
    if volatility is None:
        return 0.5
    # Clip at 100 % annualised vol; invert so that lower vol → higher score
    clipped = min(1.0, volatility)
    return 1.0 - clipped


def predict(
    sentiment_result: dict[str, Any],
    market_indicators: dict[str, Any],
) -> dict[str, Any]:
    """Generate a prediction for whether the stock will rise in 1–5 days.

    Parameters
    ----------
    sentiment_result:
        Output of :func:`analyzer.sentiment.score_headlines`, containing
        at least ``"compound"`` and ``"label"`` keys.
    market_indicators:
        Output of :func:`analyzer.market.get_market_indicators`.

    Returns
    -------
    dict[str, Any]
        - ``"probability"`` – float [0, 1] probability of upward move
        - ``"confidence_pct"`` – integer percentage (0–100)
        - ``"direction"`` – ``"rise"`` or ``"fall"``
        - ``"explanation"`` – human-readable explanation string
        - ``"recommended_stocks"`` – list with the analysed ticker data row
    """
    compound: float = sentiment_result.get("compound", 0.0)
    sentiment_label: str = sentiment_result.get("label", "neutral")
    rsi: float | None = market_indicators.get("rsi")
    price_change_pct: float | None = market_indicators.get("price_change_pct")
    sma_10: float | None = market_indicators.get("sma_10")
    sma_50: float | None = market_indicators.get("sma_50")
    volatility: float | None = market_indicators.get("volatility")

    s_comp = _sentiment_component(compound)
    m_comp = _momentum_component(rsi, price_change_pct, sma_10, sma_50)
    v_comp = _volatility_component(volatility)

    score = (
        SENTIMENT_WEIGHT * s_comp
        + MOMENTUM_WEIGHT * m_comp
        + VOLATILITY_WEIGHT * v_comp
    )
    probability = float(score)
    confidence_pct = int(round(probability * 100))
    direction = "rise" if probability >= 0.5 else "fall"

    # Build human-readable explanation
    parts: list[str] = []
    if sentiment_label == "bullish":
        parts.append("positive news sentiment")
    elif sentiment_label == "bearish":
        parts.append("negative news sentiment")

    if rsi is not None:
        if rsi > 60:
            parts.append("strong upward RSI momentum")
        elif rsi < 40:
            parts.append("weak RSI (oversold territory)")

    if price_change_pct is not None:
        if price_change_pct > 2:
            parts.append(f"recent price gain of {price_change_pct:.1f}%")
        elif price_change_pct < -2:
            parts.append(f"recent price decline of {price_change_pct:.1f}%")

    if sma_10 is not None and sma_50 is not None:
        if sma_10 > sma_50:
            parts.append("short-term SMA above long-term SMA (golden cross)")
        else:
            parts.append("short-term SMA below long-term SMA (death cross)")

    if volatility is not None:
        if volatility > 0.4:
            parts.append("high volatility environment")
        elif volatility < 0.15:
            parts.append("low volatility environment")

    if parts:
        reason = ", ".join(parts)
        if direction == "rise":
            explanation = f"Likely to rise due to {reason}."
        else:
            explanation = f"Likely to fall due to {reason}."
    else:
        explanation = "Insufficient data for a confident prediction."

    logger.info(
        "Prediction: probability=%.3f direction=%s confidence=%d%%",
        probability,
        direction,
        confidence_pct,
    )

    return {
        "probability": probability,
        "confidence_pct": confidence_pct,
        "direction": direction,
        "explanation": explanation,
    }
