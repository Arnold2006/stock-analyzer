"""Stock Analyzer – Gradio dashboard.

Entry point for the application.  Run with::

    python app.py

The Gradio interface is available at http://localhost:7860 by default.
"""

from __future__ import annotations

import io
import logging
import sys
from typing import Any

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core analysis pipeline
# ---------------------------------------------------------------------------


def run_analysis(ticker: str) -> tuple[Any, str, str, str, str, Any]:
    """Run the full analysis pipeline for *ticker* and return Gradio outputs.

    Parameters
    ----------
    ticker:
        Stock ticker symbol entered by the user.

    Returns
    -------
    tuple
        ``(recommendations_df, sentiment_html, headlines_html,
        explanation_text, confidence_html, price_chart_figure)``
    """
    ticker = ticker.strip().upper()
    if not ticker:
        empty = pd.DataFrame(columns=["Ticker", "Last Close", "5d Change %", "RSI", "Direction"])
        return empty, "—", "No ticker entered.", "Please enter a ticker symbol.", "0%", None

    logger.info("Starting analysis for %s", ticker)

    # --- imports here to keep startup fast --------------------------------
    from analyzer.market import get_market_indicators
    from analyzer.news import get_news
    from analyzer.predictor import predict
    from analyzer.sentiment import score_headlines

    # 1. Fetch data
    headlines = get_news(ticker)
    market = get_market_indicators(ticker)

    # 2. Sentiment
    sentiment = score_headlines(headlines)
    label = sentiment["label"]
    compound = sentiment["compound"]

    # 3. Prediction
    prediction = predict(sentiment, market)

    # 4. Build outputs -------------------------------------------------------

    # Recommendations table
    last_close = market.get("last_close")
    price_change_pct = market.get("price_change_pct")
    rsi = market.get("rsi")
    direction = prediction["direction"].capitalize()

    rec_df = pd.DataFrame(
        [
            {
                "Ticker": ticker,
                "Last Close": f"${last_close:.2f}" if last_close is not None else "N/A",
                "5d Change %": f"{price_change_pct:.1f}%" if price_change_pct is not None else "N/A",
                "RSI": f"{rsi:.1f}" if rsi is not None else "N/A",
                "Direction": direction,
                "Confidence": f"{prediction['confidence_pct']}%",
            }
        ]
    )

    # Sentiment indicator
    _sentiment_colors = {"bullish": "#00c853", "neutral": "#ffd600", "bearish": "#d50000"}
    color = _sentiment_colors.get(label, "#9e9e9e")
    sentiment_html = (
        f'<div style="font-size:1.4rem;font-weight:bold;color:{color};padding:8px;">'
        f"{label.upper()}"
        f'<span style="font-size:0.9rem;color:#555;margin-left:12px;">'
        f"(score: {compound:.3f})</span></div>"
    )

    # News headlines
    if headlines:
        items = "".join(
            f'<li style="margin-bottom:6px;">'
            f'<a href="{h["link"]}" target="_blank" style="color:#1565c0;">{h["title"]}</a>'
            f'<span style="color:#888;font-size:0.8rem;margin-left:8px;">{h.get("published","")}</span>'
            f"</li>"
            for h in headlines
        )
        headlines_html = f'<ul style="padding-left:20px;">{items}</ul>'
    else:
        headlines_html = "<p>No news found for this ticker.</p>"

    # Explanation
    explanation = prediction["explanation"]

    # Confidence bar
    conf = prediction["confidence_pct"]
    bar_color = "#00c853" if direction == "Rise" else "#d50000"
    confidence_html = (
        f'<div style="margin:4px 0;">'
        f'<div style="background:#e0e0e0;border-radius:4px;height:24px;width:100%;">'
        f'<div style="background:{bar_color};width:{conf}%;height:100%;border-radius:4px;'
        f'display:flex;align-items:center;justify-content:center;color:#fff;font-weight:bold;">'
        f"{conf}%</div></div></div>"
    )

    # Sparkline / price chart
    fig = _build_price_chart(ticker, market.get("df"), market.get("sma_10"), market.get("sma_50"))

    logger.info("Analysis complete for %s", ticker)
    return rec_df, sentiment_html, headlines_html, explanation, confidence_html, fig


def _build_price_chart(
    ticker: str,
    df: pd.DataFrame | None,
    sma_10: float | None,
    sma_50: float | None,
) -> Any:
    """Create a Matplotlib price + SMA sparkline figure.

    Parameters
    ----------
    ticker:
        Ticker symbol (used in title).
    df:
        Price DataFrame with a ``Close`` column.
    sma_10:
        Last 10-day SMA value (for annotation).
    sma_50:
        Last 50-day SMA value (for annotation).

    Returns
    -------
    matplotlib.figure.Figure | None
        Figure object, or ``None`` if no data available.
    """
    if df is None or df.empty:
        return None

    close = df["Close"].dropna()
    if close.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(close.index, close.values, linewidth=1.5, color="#1565c0", label="Close")

    if len(close) >= 10:
        sma10_series = close.rolling(10).mean()
        ax.plot(sma10_series.index, sma10_series.values, linewidth=1, color="#00c853",
                linestyle="--", label="SMA 10")

    if len(close) >= 50:
        sma50_series = close.rolling(50).mean()
        ax.plot(sma50_series.index, sma50_series.values, linewidth=1, color="#ff6f00",
                linestyle="--", label="SMA 50")

    ax.set_title(f"{ticker} – Price History", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio :class:`gr.Blocks` application.

    Returns
    -------
    gr.Blocks
        The assembled Gradio application (not yet launched).
    """
    with gr.Blocks(title="Stock Analyzer") as demo:
        gr.Markdown(
            "# 📈 Stock Analyzer\n"
            "Enter a ticker symbol and click **Analyze** to get a full market analysis."
        )

        with gr.Row():
            ticker_input = gr.Textbox(
                label="Ticker Symbol",
                placeholder="e.g. AAPL, TSLA, MSFT",
                scale=4,
            )
            analyze_btn = gr.Button("Analyze", variant="primary", scale=1)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Recommendation")
                rec_table = gr.Dataframe(
                    headers=["Ticker", "Last Close", "5d Change %", "RSI", "Direction", "Confidence"],
                    label="Analysis Results",
                )

                gr.Markdown("### 🎯 Confidence Score")
                confidence_output = gr.HTML(label="Confidence")

                gr.Markdown("### 🧭 Sentiment")
                sentiment_output = gr.HTML(label="Sentiment Indicator")

            with gr.Column(scale=3):
                gr.Markdown("### 📉 Price Chart")
                price_chart = gr.Plot(label="Sparkline Chart")

        gr.Markdown("### 💡 Prediction Explanation")
        explanation_output = gr.Textbox(label="Explanation", lines=3, interactive=False)

        gr.Markdown("### 📰 News Headlines")
        headlines_output = gr.HTML(label="Top News")

        analyze_btn.click(
            fn=run_analysis,
            inputs=[ticker_input],
            outputs=[
                rec_table,
                sentiment_output,
                headlines_output,
                explanation_output,
                confidence_output,
                price_chart,
            ],
        )

        ticker_input.submit(
            fn=run_analysis,
            inputs=[ticker_input],
            outputs=[
                rec_table,
                sentiment_output,
                headlines_output,
                explanation_output,
                confidence_output,
                price_chart,
            ],
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
