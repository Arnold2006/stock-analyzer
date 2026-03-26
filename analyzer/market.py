"""Market data module.

Downloads historical price data via *yfinance* and computes technical
indicators (moving averages, RSI, volatility).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from utils.cache import cached

import re

logger = logging.getLogger(__name__)

_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")


def _validate_ticker(ticker: str) -> str:
    """Validate and sanitise *ticker*, raising :exc:`ValueError` if invalid.

    Parameters
    ----------
    ticker:
        Raw ticker symbol.

    Returns
    -------
    str
        Upper-cased, stripped ticker.

    Raises
    ------
    ValueError
        If *ticker* does not match the expected pattern.
    """
    clean = ticker.strip().upper()
    if not _TICKER_RE.match(clean):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    return clean


@cached(ttl=300)
def get_price_history(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """Download OHLCV history for *ticker*.

    Parameters
    ----------
    ticker:
        Stock ticker symbol.
    period:
        History period accepted by yfinance (e.g. ``"3mo"``, ``"1y"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``,
        ``Volume``; indexed by date.  Returns an empty DataFrame on failure.
    """
    try:
        clean = _validate_ticker(ticker)
        df = yf.download(clean, period=period, auto_adjust=True, progress=False)
        # Newer yfinance versions return a MultiIndex DataFrame for single-ticker
        # downloads (columns like ('Close', 'AAPL')).  Flatten to a plain Index so
        # the rest of the code can access df["Close"] as a Series.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            logger.warning("No price data returned for %s", ticker)
        else:
            logger.info("Downloaded %d rows of price data for %s", len(df), ticker)
        return df
    except Exception as exc:
        logger.error("Failed to download price data for %s: %s", ticker, exc)
        return pd.DataFrame()


def compute_moving_averages(df: pd.DataFrame) -> dict[str, float | None]:
    """Compute 10-day and 50-day simple moving averages from *df*.

    Parameters
    ----------
    df:
        Price DataFrame with a ``Close`` column.

    Returns
    -------
    dict[str, float | None]
        Keys ``"sma_10"`` and ``"sma_50"``.
    """
    close = df["Close"].dropna() if not df.empty else pd.Series([], dtype=float)

    def _sma(window: int) -> float | None:
        if len(close) >= window:
            return float(close.rolling(window).mean().iloc[-1])
        return None

    return {"sma_10": _sma(10), "sma_50": _sma(50)}


def compute_rsi(df: pd.DataFrame, period: int = 14) -> float | None:
    """Compute the Relative Strength Index (RSI) from closing prices.

    Parameters
    ----------
    df:
        Price DataFrame with a ``Close`` column.
    period:
        Look-back window for RSI calculation (default: 14).

    Returns
    -------
    float | None
        Most recent RSI value (0–100), or ``None`` if insufficient data.
    """
    close = df["Close"].dropna() if not df.empty else pd.Series([], dtype=float)
    if len(close) < period + 1:
        return None
    delta = close.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def compute_volatility(df: pd.DataFrame, window: int = 10) -> float | None:
    """Compute annualised close-to-close volatility.

    Parameters
    ----------
    df:
        Price DataFrame with a ``Close`` column.
    window:
        Rolling window in trading days (default: 10).

    Returns
    -------
    float | None
        Annualised volatility (fraction), or ``None`` if insufficient data.
    """
    close = df["Close"].dropna() if not df.empty else pd.Series([], dtype=float)
    if len(close) < window + 1:
        return None
    log_returns = np.log(close / close.shift(1)).dropna()
    vol = float(log_returns.rolling(window).std().iloc[-1]) * np.sqrt(252)
    return vol


def get_market_indicators(ticker: str) -> dict[str, Any]:
    """Fetch price history and compute all market indicators for *ticker*.

    Parameters
    ----------
    ticker:
        Stock ticker symbol.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``"df"`` – raw price DataFrame
        - ``"sma_10"`` / ``"sma_50"`` – moving averages
        - ``"rsi"`` – RSI value
        - ``"volatility"`` – annualised volatility
        - ``"last_close"`` – most recent closing price
        - ``"price_change_pct"`` – percentage change over last 5 days
    """
    df = get_price_history(ticker)
    if df.empty:
        return {
            "df": df,
            "sma_10": None,
            "sma_50": None,
            "rsi": None,
            "volatility": None,
            "last_close": None,
            "price_change_pct": None,
        }

    mas = compute_moving_averages(df)
    rsi = compute_rsi(df)
    vol = compute_volatility(df)

    close = df["Close"].dropna()
    last_close = float(close.iloc[-1]) if len(close) >= 1 else None
    price_change_pct: float | None = None
    if len(close) >= 6:
        price_change_pct = float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100)

    return {
        "df": df,
        "sma_10": mas["sma_10"],
        "sma_50": mas["sma_50"],
        "rsi": rsi,
        "volatility": vol,
        "last_close": last_close,
        "price_change_pct": price_change_pct,
    }
