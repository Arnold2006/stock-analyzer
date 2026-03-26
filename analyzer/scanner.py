"""Scandinavian stock scanner.

Scans major Scandinavian stock exchanges for day trading opportunities
based on technical indicators (RSI, volatility, volume ratio, momentum).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from analyzer.market import get_market_indicators

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated list of Scandinavian stocks across four major exchanges
# ---------------------------------------------------------------------------

SCANDINAVIAN_STOCKS: list[dict[str, str]] = [
    # --- Stockholm Stock Exchange (Nasdaq Stockholm) — suffix .ST ---
    {"ticker": "ERIC-B.ST", "name": "Ericsson B", "exchange": "Stockholm"},
    {"ticker": "VOLV-B.ST", "name": "Volvo B", "exchange": "Stockholm"},
    {"ticker": "ABB.ST", "name": "ABB", "exchange": "Stockholm"},
    {"ticker": "SWED-A.ST", "name": "Swedbank A", "exchange": "Stockholm"},
    {"ticker": "SEB-A.ST", "name": "SEB A", "exchange": "Stockholm"},
    {"ticker": "ATCO-A.ST", "name": "Atlas Copco A", "exchange": "Stockholm"},
    {"ticker": "SAND.ST", "name": "Sandvik", "exchange": "Stockholm"},
    {"ticker": "SKF-B.ST", "name": "SKF B", "exchange": "Stockholm"},
    {"ticker": "HEXA-B.ST", "name": "Hexagon B", "exchange": "Stockholm"},
    {"ticker": "ESSITY-B.ST", "name": "Essity B", "exchange": "Stockholm"},
    {"ticker": "ALFA.ST", "name": "Alfa Laval", "exchange": "Stockholm"},
    {"ticker": "NIBE-B.ST", "name": "NIBE Industrier B", "exchange": "Stockholm"},
    {"ticker": "INVE-B.ST", "name": "Investor B", "exchange": "Stockholm"},
    {"ticker": "SSAB-A.ST", "name": "SSAB A", "exchange": "Stockholm"},
    {"ticker": "NDA-SE.ST", "name": "Nordea Bank", "exchange": "Stockholm"},
    # --- Oslo Stock Exchange (Oslo Børs) — suffix .OL ---
    {"ticker": "EQNR.OL", "name": "Equinor", "exchange": "Oslo"},
    {"ticker": "NHY.OL", "name": "Norsk Hydro", "exchange": "Oslo"},
    {"ticker": "MOWI.OL", "name": "Mowi", "exchange": "Oslo"},
    {"ticker": "DNB.OL", "name": "DNB Bank", "exchange": "Oslo"},
    {"ticker": "ORK.OL", "name": "Orkla", "exchange": "Oslo"},
    {"ticker": "TEL.OL", "name": "Telenor", "exchange": "Oslo"},
    {"ticker": "YARA.OL", "name": "Yara International", "exchange": "Oslo"},
    {"ticker": "TOM.OL", "name": "Tomra Systems", "exchange": "Oslo"},
    {"ticker": "AKSO.OL", "name": "Aker Solutions", "exchange": "Oslo"},
    {"ticker": "RECSI.OL", "name": "REC Silicon", "exchange": "Oslo"},
    # --- Copenhagen Stock Exchange (Nasdaq Copenhagen) — suffix .CO ---
    {"ticker": "NOVO-B.CO", "name": "Novo Nordisk B", "exchange": "Copenhagen"},
    {"ticker": "MAERSK-B.CO", "name": "A.P. Møller-Mærsk B", "exchange": "Copenhagen"},
    {"ticker": "CARL-B.CO", "name": "Carlsberg B", "exchange": "Copenhagen"},
    {"ticker": "ORSTED.CO", "name": "Ørsted", "exchange": "Copenhagen"},
    {"ticker": "VWS.CO", "name": "Vestas Wind Systems", "exchange": "Copenhagen"},
    {"ticker": "DSV.CO", "name": "DSV", "exchange": "Copenhagen"},
    {"ticker": "COLO-B.CO", "name": "Coloplast B", "exchange": "Copenhagen"},
    {"ticker": "GMAB.CO", "name": "Genmab", "exchange": "Copenhagen"},
    # --- Helsinki Stock Exchange (Nasdaq Helsinki) — suffix .HE ---
    {"ticker": "NOKIA.HE", "name": "Nokia", "exchange": "Helsinki"},
    {"ticker": "SAMPO.HE", "name": "Sampo", "exchange": "Helsinki"},
    {"ticker": "FORTUM.HE", "name": "Fortum", "exchange": "Helsinki"},
    {"ticker": "NESTE.HE", "name": "Neste", "exchange": "Helsinki"},
    {"ticker": "STERV.HE", "name": "Stora Enso R", "exchange": "Helsinki"},
    {"ticker": "UPM.HE", "name": "UPM-Kymmene", "exchange": "Helsinki"},
    {"ticker": "WRT1V.HE", "name": "Wärtsilä", "exchange": "Helsinki"},
    {"ticker": "METSO.HE", "name": "Metso Outotec", "exchange": "Helsinki"},
]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _compute_volume_ratio(df: pd.DataFrame, window: int = 20) -> float | None:
    """Return the ratio of the latest day's volume to the rolling average.

    Parameters
    ----------
    df:
        Price DataFrame that includes a ``Volume`` column.
    window:
        Rolling window in trading days for the average.

    Returns
    -------
    float | None
        Volume ratio (>1 means above-average activity), or ``None`` if the
        data is insufficient.
    """
    if df.empty or "Volume" not in df.columns:
        return None
    volume = df["Volume"].dropna()
    if len(volume) < 2:
        return None
    avg = float(volume.rolling(min(window, len(volume))).mean().iloc[-1])
    if avg == 0:
        return None
    return float(volume.iloc[-1] / avg)


def _compute_day_trading_score(
    volume_ratio: float | None,
    volatility: float | None,
    rsi: float | None,
    price_change_pct: float | None,
) -> float:
    """Compute a normalised day trading opportunity score in [0, 1].

    A higher score indicates that a stock shows characteristics attractive
    to day traders: high relative volume, meaningful intraday volatility,
    an extreme RSI reading, and notable short-term price momentum.

    Weighting
    ---------
    * Volume ratio  – 30 %  (liquidity / unusual activity)
    * Volatility    – 30 %  (price movement potential)
    * RSI extremity – 20 %  (overbought/oversold momentum signal)
    * Momentum      – 20 %  (absolute 5-day price change)

    Parameters
    ----------
    volume_ratio:
        Ratio of latest volume to the rolling 20-day average.
    volatility:
        Annualised volatility (fraction).
    rsi:
        14-day RSI value (0–100).
    price_change_pct:
        5-day price change in percent.

    Returns
    -------
    float
        Score in [0, 1].
    """
    components: list[tuple[float, float]] = []  # (score, weight)

    if volume_ratio is not None:
        # Normalise: a ratio of 3× or more maps to 1.0
        components.append((min(volume_ratio / 3.0, 1.0), 0.30))

    if volatility is not None:
        # Normalise: annualised vol of 50 % or more maps to 1.0
        components.append((min(volatility / 0.50, 1.0), 0.30))

    if rsi is not None:
        # How far from neutral (50)? 0 or 100 → 1.0, 50 → 0.0
        components.append((abs(rsi - 50.0) / 50.0, 0.20))

    if price_change_pct is not None:
        # Normalise: ±5 % or more maps to 1.0
        components.append((min(abs(price_change_pct) / 5.0, 1.0), 0.20))

    if not components:
        return 0.0

    total_weight = sum(w for _, w in components)
    weighted_sum = sum(s * w for s, w in components)
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Per-ticker scanning
# ---------------------------------------------------------------------------


def _scan_single(stock: dict[str, str]) -> dict[str, Any] | None:
    """Fetch data for one stock and compute its day trading metrics.

    Parameters
    ----------
    stock:
        Entry from :data:`SCANDINAVIAN_STOCKS` with keys ``"ticker"``,
        ``"name"``, and ``"exchange"``.

    Returns
    -------
    dict | None
        Metrics dict, or ``None`` if data could not be retrieved.
    """
    ticker = stock["ticker"]
    try:
        market = get_market_indicators(ticker)
        if market["last_close"] is None:
            logger.debug("No data for %s, skipping", ticker)
            return None

        volume_ratio = _compute_volume_ratio(market["df"])
        score = _compute_day_trading_score(
            volume_ratio=volume_ratio,
            volatility=market["volatility"],
            rsi=market["rsi"],
            price_change_pct=market["price_change_pct"],
        )

        return {
            "Ticker": ticker,
            "Company": stock["name"],
            "Exchange": stock["exchange"],
            "_last_close": market["last_close"],
            "_price_change_pct": market["price_change_pct"],
            "_volume_ratio": volume_ratio,
            "_rsi": market["rsi"],
            "_volatility": market["volatility"],
            "_score": score,
        }
    except Exception as exc:
        logger.error("Error scanning %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_scandinavian_stocks(top_n: int = 20) -> pd.DataFrame:
    """Scan Scandinavian exchanges and return top day trading opportunities.

    Fetches live market data for all stocks in :data:`SCANDINAVIAN_STOCKS`
    concurrently, computes a day trading score for each, and returns the
    top *top_n* ranked results as a formatted :class:`pandas.DataFrame`.

    Parameters
    ----------
    top_n:
        Maximum number of stocks to return (sorted by score, descending).

    Returns
    -------
    pd.DataFrame
        Columns: ``Ticker``, ``Company``, ``Exchange``, ``Last Close``,
        ``5d Change %``, ``Vol Ratio``, ``RSI``, ``Volatility``, ``Score``.
        Returns an empty DataFrame (with the same columns) if no data was
        available.
    """
    logger.info("Starting Scandinavian stock scan (%d stocks)", len(SCANDINAVIAN_STOCKS))
    raw_results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_scan_single, s): s for s in SCANDINAVIAN_STOCKS}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                raw_results.append(result)

    _cols = ["Ticker", "Company", "Exchange", "Last Close", "5d Change %",
             "Vol Ratio", "RSI", "Volatility", "Score"]

    if not raw_results:
        return pd.DataFrame(columns=_cols)

    df = pd.DataFrame(raw_results)
    df = df.sort_values("_score", ascending=False).head(top_n).reset_index(drop=True)

    # Build display DataFrame with formatted strings
    display = pd.DataFrame()
    display["Ticker"] = df["Ticker"]
    display["Company"] = df["Company"]
    display["Exchange"] = df["Exchange"]
    display["Last Close"] = df["_last_close"].apply(
        lambda x: f"{x:.2f}" if x is not None else "N/A"
    )
    display["5d Change %"] = df["_price_change_pct"].apply(
        lambda x: f"{x:+.1f}%" if x is not None else "N/A"
    )
    display["Vol Ratio"] = df["_volume_ratio"].apply(
        lambda x: f"{x:.2f}×" if x is not None else "N/A"
    )
    display["RSI"] = df["_rsi"].apply(
        lambda x: f"{x:.1f}" if x is not None else "N/A"
    )
    display["Volatility"] = df["_volatility"].apply(
        lambda x: f"{x:.1%}" if x is not None else "N/A"
    )
    display["Score"] = df["_score"].apply(lambda x: f"{x:.2f}")

    logger.info("Scan complete — %d opportunities returned", len(display))
    return display
