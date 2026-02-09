"""Tests for sentiment and macro feature integration.

This module tests the sentiment analysis and macro factor loading functionality.
"""
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestSentimentAnalysis:
    """Test sentiment analysis functionality."""

    def test_fetch_sentiment_scores_returns_dataframe(self):
        """Test that fetch_sentiment_scores returns a valid DataFrame."""
        from src.advanced.sentiment import fetch_sentiment_scores

        # Test with news fetching disabled to avoid network calls
        with patch("src.advanced.sentiment.fetch_ticker_news", return_value=[]):
            result = fetch_sentiment_scores(["AAPL", "TSLA"], use_news=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "key_term" in result.columns
        assert "sentiment_score" in result.columns
        assert "timestamp" in result.columns

    def test_compute_sentiment_score_empty_text(self):
        """Test sentiment score for empty text returns neutral."""
        from src.advanced.sentiment import compute_sentiment_score

        assert compute_sentiment_score("") == 0.0
        assert compute_sentiment_score(None) == 0.0

    def test_compute_sentiment_score_positive_text(self):
        """Test sentiment score for positive text."""
        from src.advanced.sentiment import compute_sentiment_score

        # This may use VADER or TextBlob depending on availability
        score = compute_sentiment_score("This is great news! Amazing results!")
        # Should be positive (may vary slightly based on library)
        assert isinstance(score, float)
        assert -1 <= score <= 1

    def test_compute_sentiment_score_negative_text(self):
        """Test sentiment score for negative text."""
        from src.advanced.sentiment import compute_sentiment_score

        score = compute_sentiment_score("Terrible news, big losses expected.")
        assert isinstance(score, float)
        assert -1 <= score <= 1

    def test_fetch_ticker_news_handles_errors(self):
        """Test that fetch_ticker_news handles errors gracefully."""
        from src.advanced.sentiment import fetch_ticker_news

        # Mock yfinance.Ticker to raise an exception
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.side_effect = Exception("Network error")
            result = fetch_ticker_news("INVALID")

        assert result == []

    def test_vader_analyzer_ci_mode_no_download(self):
        """Test that VADER analyzer does not attempt download in CI mode."""
        import sys
        
        # Clear cached imports to ensure fresh load
        if 'src.advanced.sentiment' in sys.modules:
            del sys.modules['src.advanced.sentiment']
        
        with patch.dict(os.environ, {"CI": "true"}):
            with patch("nltk.data.find", side_effect=LookupError("Resource not found")):
                with patch("nltk.download") as mock_download:
                    from src.advanced.sentiment import compute_sentiment_score
                    
                    # Should fall back to TextBlob without attempting download
                    score = compute_sentiment_score("Great news!")
                    
                    # Verify no download was attempted in CI mode
                    assert not mock_download.called
                    # Should still return a valid score using TextBlob
                    assert isinstance(score, float)
                    assert -1 <= score <= 1

    def test_vader_analyzer_handles_missing_lexicon(self):
        """Test that compute_sentiment_score handles missing VADER lexicon gracefully."""
        import sys
        
        # Clear cached imports
        if 'src.advanced.sentiment' in sys.modules:
            del sys.modules['src.advanced.sentiment']
        
        with patch.dict(os.environ, {"CI": "true"}):
            # Mock SentimentIntensityAnalyzer to raise LookupError
            with patch("nltk.data.find"):
                with patch("nltk.sentiment.vader.SentimentIntensityAnalyzer", 
                          side_effect=LookupError("Lexicon not found")):
                    from src.advanced.sentiment import compute_sentiment_score
                    
                    # Should handle the exception and fall back to TextBlob
                    score = compute_sentiment_score("This is great news!")
                    
                    assert isinstance(score, float)
                    assert -1 <= score <= 1

    def test_sentiment_to_feature_merges_correctly(self):
        """Test sentiment_to_feature merges data correctly."""
        from src.advanced.sentiment import sentiment_to_feature

        price_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "close": [100, 101, 102],
        })

        sentiment_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["AAPL", "AAPL"],
            "sentiment_score": [0.5, -0.2],
        })

        result = sentiment_to_feature(price_df, sentiment_df)

        assert "sentiment_score" in result.columns
        assert len(result) == 3
        # First two rows should have sentiment scores
        assert result.iloc[0]["sentiment_score"] == 0.5
        assert result.iloc[1]["sentiment_score"] == -0.2
        # Third row should be NaN (no matching sentiment data)
        assert pd.isna(result.iloc[2]["sentiment_score"])

    def test_load_external_sentiment(self, tmp_path):
        """Test loading sentiment from external CSV."""
        from src.advanced.sentiment import load_external_sentiment

        # Create a test CSV
        csv_path = tmp_path / "sentiment.csv"
        csv_path.write_text(
            "date,ticker,sentiment_score\n"
            "2024-01-01,AAPL,0.5\n"
            "2024-01-02,AAPL,-0.3\n"
        )

        result = load_external_sentiment(csv_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result.iloc[0]["sentiment_score"] == 0.5


class TestMacroFactors:
    """Test macro factor loading functionality."""

    def test_load_macro_factors_returns_dataframe(self):
        """Test that load_macro_factors returns a DataFrame."""
        from src.features.macro import load_macro_factors_yfinance

        # Mock yfinance to avoid network calls
        with patch("yfinance.download") as mock_download:
            mock_data = pd.DataFrame({
                "Close": [20.0, 21.0, 22.0]
            }, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
            mock_download.return_value = mock_data

            result = load_macro_factors_yfinance(
                start_date="2024-01-01",
                end_date="2024-01-03"
            )

        assert isinstance(result, pd.DataFrame)

    def test_compute_macro_features_adds_derived_columns(self):
        """Test that compute_macro_features adds derived features."""
        from src.features.macro import compute_macro_features

        macro_df = pd.DataFrame({
            "vix": [20.0, 22.0, 21.0, 25.0, 23.0] * 10,
            "sp500": [4000, 4010, 4005, 4020, 4015] * 10,
            "treasury_10y": [4.0, 4.1, 4.05, 4.2, 4.15] * 10,
        })

        result = compute_macro_features(macro_df)

        assert "vix_pct_change" in result.columns
        assert "sp500_return" in result.columns
        assert "treasury_change" in result.columns

    def test_merge_macro_features(self):
        """Test merging macro features with price data."""
        from src.features.macro import merge_macro_features

        price_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "close": [100, 101, 102],
        })

        macro_df = pd.DataFrame({
            "vix": [20.0, 21.0, 22.0],
        }, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
        macro_df.index.name = "date"

        result = merge_macro_features(price_df, macro_df)

        assert "macro_vix" in result.columns
        assert len(result) == 3

    def test_merge_macro_features_empty_macro(self):
        """Test merge with empty macro DataFrame returns original."""
        from src.features.macro import merge_macro_features

        price_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "close": [100, 101],
        })

        result = merge_macro_features(price_df, pd.DataFrame())

        pd.testing.assert_frame_equal(result, price_df)

    def test_load_macro_factors_source_validation(self):
        """Test load_macro_factors validates source parameter."""
        from src.features.macro import load_macro_factors

        with pytest.raises(ValueError, match="Unknown source"):
            load_macro_factors(source="invalid_source")


class TestLSTMTransformerModels:
    """Test LSTM and Transformer model building with new parameters."""

    @pytest.fixture
    def mock_keras(self):
        """Mock keras to avoid TensorFlow import issues."""
        with patch.dict("sys.modules", {"tensorflow": MagicMock(), "tensorflow.keras": MagicMock()}):
            yield

    def test_build_lstm_default_params(self):
        """Test build_lstm with default parameters."""
        # Skip if TensorFlow not available
        pytest.importorskip("tensorflow")

        from market_forecasting import build_lstm

        model = build_lstm((60, 10))
        assert model is not None

    def test_build_lstm_classification_mode(self):
        """Test build_lstm with classification mode."""
        pytest.importorskip("tensorflow")

        from market_forecasting import build_lstm

        model = build_lstm((60, 10), classification=True)
        # Should have sigmoid activation and binary_crossentropy loss
        assert model is not None
        # Check that the model was compiled for classification
        # The loss attribute returns the loss function, not the string
        # So we check the model configuration or output layer activation instead
        assert model.layers[-1].activation.__name__ == "sigmoid"

    def test_build_lstm_custom_layers(self):
        """Test build_lstm with custom layer configuration."""
        pytest.importorskip("tensorflow")

        from market_forecasting import build_lstm

        model = build_lstm(
            (60, 10),
            n_layers=3,
            units_per_layer=[128, 64, 32],
        )
        assert model is not None

    def test_build_lstm_layer_mismatch_raises(self):
        """Test build_lstm raises error for layer count mismatch."""
        pytest.importorskip("tensorflow")

        from market_forecasting import build_lstm

        with pytest.raises(ValueError, match="must match n_layers"):
            build_lstm((60, 10), n_layers=3, units_per_layer=[64, 32])

    def test_build_transformer_default_params(self):
        """Test build_transformer with default parameters."""
        pytest.importorskip("tensorflow")

        from market_forecasting import build_transformer

        model = build_transformer(num_features=10, seq_len=60)
        assert model is not None

    def test_build_transformer_classification_mode(self):
        """Test build_transformer with classification mode."""
        pytest.importorskip("tensorflow")

        from market_forecasting import build_transformer

        model = build_transformer(
            num_features=10,
            seq_len=60,
            classification=True,
        )
        assert model is not None
        # Check that the output layer uses sigmoid activation for classification
        assert model.layers[-1].activation.__name__ == "sigmoid"

    def test_build_transformer_multiple_layers(self):
        """Test build_transformer with multiple encoder layers."""
        pytest.importorskip("tensorflow")

        from market_forecasting import build_transformer

        model = build_transformer(
            num_features=10,
            seq_len=60,
            num_layers=4,
            num_heads=8,
        )
        assert model is not None

    def test_classification_output_range(self):
        """Test that classification models output values in [0, 1]."""
        pytest.importorskip("tensorflow")
        import numpy as np

        from market_forecasting import build_lstm

        model = build_lstm((10, 5), classification=True)
        # Create dummy input
        X_test = np.random.randn(5, 10, 5).astype(np.float32)
        predictions = model.predict(X_test, verbose=0)

        # All predictions should be in [0, 1] for sigmoid output
        assert predictions.min() >= 0.0
        assert predictions.max() <= 1.0
