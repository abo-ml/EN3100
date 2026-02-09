"""Sentiment analysis integration for market forecasting.

This module provides sentiment scoring functionality using VADER (Valence Aware
Dictionary and sEntiment Reasoner) from NLTK or TextBlob as a fallback. It can
fetch headlines from yfinance news or accept externally prepared sentiment data.

Environment Variables:
- NEWSAPI_KEY: Optional API key for NewsAPI integration (future enhancement).
  When implementing NewsAPI support, use `check_env_var("NEWSAPI_KEY")` from
  `src.utils` to validate the key is set before making API calls.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# Note: When implementing NewsAPI integration, use:
#   from ..utils import check_env_var
#   api_key = check_env_var("NEWSAPI_KEY", warn_if_missing=True)

logger = logging.getLogger(__name__)


def _get_vader_analyzer():
    """Get VADER sentiment analyzer, downloading lexicon if needed.

    Returns
    -------
    SentimentIntensityAnalyzer or None
        VADER analyzer if available, None otherwise.
    """
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            logger.info("Downloading VADER lexicon...")
            nltk.download("vader_lexicon", quiet=True)

        return SentimentIntensityAnalyzer()
    except ImportError:
        logger.warning("NLTK not installed. VADER sentiment unavailable.")
        return None


def _get_textblob_sentiment(text: str) -> float:
    """Get sentiment polarity using TextBlob.

    Parameters
    ----------
    text : str
        Text to analyze.

    Returns
    -------
    float
        Polarity score between -1 (negative) and 1 (positive).
    """
    try:
        from textblob import TextBlob

        blob = TextBlob(text)
        return blob.sentiment.polarity
    except ImportError:
        logger.warning("TextBlob not installed. Returning neutral sentiment.")
        return 0.0


def compute_sentiment_score(text: str, use_vader: bool = True) -> float:
    """Compute sentiment score for a piece of text.

    Parameters
    ----------
    text : str
        Text to analyze (headline, article, etc.).
    use_vader : bool
        If True, try VADER first, fall back to TextBlob. Default is True.

    Returns
    -------
    float
        Sentiment score between -1 (negative) and 1 (positive).
    """
    if not text or not isinstance(text, str):
        return 0.0

    if use_vader:
        analyzer = _get_vader_analyzer()
        if analyzer:
            scores = analyzer.polarity_scores(text)
            # compound score is normalized between -1 and 1
            return scores["compound"]

    # Fallback to TextBlob
    return _get_textblob_sentiment(text)


def fetch_ticker_news(ticker: str, max_headlines: int = 10) -> List[str]:
    """Fetch recent news headlines for a ticker using yfinance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    max_headlines : int
        Maximum number of headlines to fetch. Default is 10.

    Returns
    -------
    List[str]
        List of headline strings.
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return []

        headlines = []
        for item in news[:max_headlines]:
            title = item.get("title", "")
            if title:
                headlines.append(title)

        return headlines
    except Exception as e:
        logger.warning(f"Could not fetch news for {ticker}: {e}")
        return []


def fetch_sentiment_scores(
    key_terms: Iterable[str],
    use_news: bool = True,
    use_vader: bool = True,
) -> pd.DataFrame:
    """Fetch sentiment scores for a list of tickers or key terms.

    If `use_news` is True, attempts to fetch headlines from yfinance and compute
    sentiment from them. Otherwise, returns neutral sentiment.

    Parameters
    ----------
    key_terms : Iterable[str]
        List of ticker symbols or key terms to analyze.
    use_news : bool
        If True, fetch headlines from yfinance. Default is True.
    use_vader : bool
        If True, use VADER for sentiment analysis. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, key_term (ticker), sentiment_score
    """
    rows = []
    timestamp = pd.Timestamp.now("UTC")

    for term in key_terms:
        sentiment_score = 0.0

        if use_news:
            headlines = fetch_ticker_news(term)
            if headlines:
                # Average sentiment across all headlines
                scores = [
                    compute_sentiment_score(h, use_vader=use_vader)
                    for h in headlines
                ]
                sentiment_score = sum(scores) / len(scores) if scores else 0.0

        rows.append({
            "timestamp": timestamp,
            "key_term": term,
            "sentiment_score": sentiment_score,
        })

    return pd.DataFrame(rows)


def load_external_sentiment(csv_path: Path) -> pd.DataFrame:
    """Load sentiment scores from a CSV prepared outside the pipeline.

    Expected columns
    ----------------
    date: ISO date (YYYY-MM-DD) matching the price data frequency.
    ticker: Ticker symbol aligned to the OHLCV files.
    sentiment_score: Numeric sentiment value per (date, ticker).
    """

    sentiment_df = pd.read_csv(csv_path, parse_dates=["date"])
    sentiment_df["ticker"] = sentiment_df["ticker"].astype(str)
    sentiment_df["sentiment_score"] = pd.to_numeric(sentiment_df["sentiment_score"], errors="coerce")
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    return sentiment_df


def sentiment_to_feature(price_df: pd.DataFrame, sentiment_df: pd.DataFrame, date_key: str = "date") -> pd.DataFrame:
    """Align sentiment scores with price data (left join on date and ticker)."""

    temp = sentiment_df.copy()
    temp[date_key] = pd.to_datetime(temp[date_key])
    price_df = price_df.copy()
    price_df[date_key] = pd.to_datetime(price_df[date_key])
    merged = price_df.merge(temp[["ticker", date_key, "sentiment_score"]], on=["ticker", date_key], how="left")
    return merged
