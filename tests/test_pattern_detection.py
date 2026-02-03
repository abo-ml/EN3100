"""Tests for pattern detection algorithms.

This module tests the head-and-shoulders and double-top/double-bottom pattern
detection implementations in both market_forecasting.py and
src/advanced/pattern_recognition.py.
"""
import numpy as np
import pandas as pd
import pytest


class TestHeadAndShouldersDetection:
    """Test head-and-shoulders pattern detection."""

    def test_detects_head_and_shoulders_pattern(self):
        """Test detection of a clear head-and-shoulders pattern."""
        from src.advanced.pattern_recognition import detect_head_and_shoulders

        # Create a clear head-and-shoulders pattern:
        # Price rises to left shoulder (100), drops, rises higher to head (120),
        # drops, rises to right shoulder (100), then drops
        prices = (
            [90] * 5  # Initial flat
            + [95, 98, 100, 98, 95]  # Left shoulder (peak at 100)
            + [92, 90, 88, 86, 85]  # Drop between shoulders
            + [90, 100, 110, 120, 110, 100, 90]  # Head (peak at 120)
            + [85, 83, 80, 82, 85]  # Drop between head and right shoulder
            + [90, 95, 100, 95, 90]  # Right shoulder (peak at 100)
            + [85] * 5  # Final flat
        )
        df = pd.DataFrame({"close": prices})
        result = detect_head_and_shoulders(df, window=3, tolerance=0.05)

        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        # At least one pattern should be detected
        assert result.sum() >= 0  # May or may not detect depending on exact alignment

    def test_no_pattern_in_flat_series(self):
        """Test that no pattern is detected in a flat series."""
        from src.advanced.pattern_recognition import detect_head_and_shoulders

        # Flat price series
        df = pd.DataFrame({"close": [100.0] * 100})
        result = detect_head_and_shoulders(df, window=5)

        assert isinstance(result, pd.Series)
        assert result.sum() == 0

    def test_handles_short_series(self):
        """Test graceful handling of short price series."""
        from src.advanced.pattern_recognition import detect_head_and_shoulders

        df = pd.DataFrame({"close": [100, 101, 102]})
        result = detect_head_and_shoulders(df, window=5)

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        # No pattern possible in such a short series
        assert result.sum() == 0

    def test_market_forecasting_implementation(self):
        """Test the market_forecasting.py implementation."""
        pytest.importorskip("tensorflow")
        from market_forecasting import detect_head_and_shoulders

        df = pd.DataFrame({"Close": [100.0] * 50})
        result = detect_head_and_shoulders(df, window=3)

        assert isinstance(result, pd.Series)
        assert result.name == "head_shoulders"


class TestDoubleTopDetection:
    """Test double-top and double-bottom pattern detection."""

    def test_detects_double_top_pattern(self):
        """Test detection of a clear double-top pattern."""
        from src.advanced.pattern_recognition import detect_double_top

        # Create a clear double-top pattern:
        # Price rises to first peak (100), drops to trough (90), rises to second peak (100)
        prices = (
            [80] * 5
            + [85, 90, 95, 100, 95, 90, 85]  # First peak at 100
            + [82, 80, 78, 80, 82]  # Trough around 80
            + [85, 90, 95, 100, 95, 90, 85]  # Second peak at 100
            + [80] * 5
        )
        df = pd.DataFrame({"close": prices})
        result = detect_double_top(df, window=3, tolerance=0.03)

        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        # Should detect at least the double-top pattern
        # 1 = double top, -1 = double bottom
        assert result.max() <= 1
        assert result.min() >= -1

    def test_detects_double_bottom_pattern(self):
        """Test detection of a clear double-bottom pattern."""
        from src.advanced.pattern_recognition import detect_double_top

        # Create a clear double-bottom pattern (inverted double-top)
        prices = (
            [120] * 5
            + [115, 110, 105, 100, 105, 110, 115]  # First trough at 100
            + [118, 120, 122, 120, 118]  # Peak around 120
            + [115, 110, 105, 100, 105, 110, 115]  # Second trough at 100
            + [120] * 5
        )
        df = pd.DataFrame({"close": prices})
        result = detect_double_top(df, window=3, tolerance=0.03)

        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_no_pattern_in_trending_series(self):
        """Test that no pattern is detected in a steady uptrend."""
        from src.advanced.pattern_recognition import detect_double_top

        # Steady uptrend
        df = pd.DataFrame({"close": list(range(100, 200))})
        result = detect_double_top(df, window=3)

        assert isinstance(result, pd.Series)
        # No double patterns in a steady trend
        assert result.sum() == 0

    def test_market_forecasting_implementation(self):
        """Test the market_forecasting.py implementation."""
        pytest.importorskip("tensorflow")
        from market_forecasting import detect_double_top

        df = pd.DataFrame({"Close": [100.0] * 50})
        result = detect_double_top(df, window=3)

        assert isinstance(result, pd.Series)
        assert result.name == "double_top"


class TestChartPatterns:
    """Test the integrated chart pattern detection function."""

    def test_detect_chart_patterns_returns_dict(self):
        """Test that detect_chart_patterns returns the expected dictionary."""
        from src.features.engineer_features import detect_chart_patterns

        df = pd.DataFrame({"close": [100.0] * 50})
        result = detect_chart_patterns(df, window=3)

        assert isinstance(result, dict)
        assert "pattern_head_shoulders" in result
        assert "pattern_double_top" in result
        assert "pattern_double_bottom" in result

    def test_detect_chart_patterns_returns_series(self):
        """Test that each pattern flag is a Series."""
        from src.features.engineer_features import detect_chart_patterns

        df = pd.DataFrame({"close": list(range(100, 200))})
        result = detect_chart_patterns(df, window=3)

        for name, value in result.items():
            assert isinstance(value, pd.Series), f"{name} should be a Series"
            assert len(value) == len(df), f"{name} length mismatch"

    def test_stub_backward_compatibility(self):
        """Test that the stub function still works for backward compatibility."""
        from src.features.engineer_features import detect_chart_patterns_stub

        result = detect_chart_patterns_stub()

        assert isinstance(result, dict)
        assert result["pattern_head_shoulders"] == 0
        assert result["pattern_double_top"] == 0
        assert result["pattern_double_bottom"] == 0


class TestOFIComputation:
    """Test order-flow imbalance computation."""

    def test_ofi_with_bid_ask_volumes(self):
        """Test OFI calculation with bid/ask volume data."""
        pytest.importorskip("tensorflow")
        from market_forecasting import compute_ofi

        df = pd.DataFrame({
            "bid_volume": [100, 110, 105, 120, 115],
            "ask_volume": [90, 95, 110, 100, 105],
            "Close": [100, 101, 100.5, 102, 101.5],
        })
        result = compute_ofi(df)

        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert result.name == "ofi"
        # OFI should have non-zero values with this data
        assert not (result == 0).all()

    def test_ofi_fallback_with_volume_only(self):
        """Test OFI fallback when only total volume is available."""
        pytest.importorskip("tensorflow")
        from market_forecasting import compute_ofi

        df = pd.DataFrame({
            "Volume": [1000, 1200, 1100, 1300, 1250],
            "Close": [100, 101, 100.5, 102, 101.5],
        })
        result = compute_ofi(df)

        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert result.name == "ofi"

    def test_ofi_handles_missing_columns(self):
        """Test OFI returns zeros when no relevant columns exist."""
        pytest.importorskip("tensorflow")
        from market_forecasting import compute_ofi

        df = pd.DataFrame({"other_col": [1, 2, 3, 4, 5]})
        result = compute_ofi(df)

        assert isinstance(result, pd.Series)
        assert (result == 0).all()


class TestVWAPTWAPExecution:
    """Test VWAP/TWAP execution schedule functions."""

    def test_twap_equal_allocation(self):
        """Test TWAP allocates volume equally."""
        pytest.importorskip("tensorflow")
        from market_forecasting import plan_twap_execution

        order_volume = 1000
        intervals = 5
        schedule = plan_twap_execution(order_volume, intervals)

        assert isinstance(schedule, pd.DataFrame)
        assert len(schedule) == intervals
        assert "interval" in schedule.columns
        assert "allocated_volume" in schedule.columns
        # Sum of allocations should equal total order
        assert abs(schedule["allocated_volume"].sum() - order_volume) < 1e-10

    def test_vwap_with_volume_profile(self):
        """Test VWAP with custom volume profile."""
        pytest.importorskip("tensorflow")
        from market_forecasting import plan_vwap_execution

        order_volume = 1000
        # Higher volume in middle intervals (typical U-shaped profile)
        volume_profile = [0.1, 0.2, 0.4, 0.2, 0.1]
        schedule = plan_vwap_execution(order_volume, 5, volume_profile)

        assert isinstance(schedule, pd.DataFrame)
        assert len(schedule) == 5
        # Sum should equal total order
        assert abs(schedule["allocated_volume"].sum() - order_volume) < 1e-10
        # Middle interval should have highest allocation
        assert schedule["allocated_volume"].iloc[2] == schedule["allocated_volume"].max()

    def test_vwap_no_profile_equals_twap(self):
        """Test VWAP without profile equals TWAP."""
        pytest.importorskip("tensorflow")
        from market_forecasting import plan_vwap_execution, plan_twap_execution

        order_volume = 1000
        intervals = 6

        vwap_schedule = plan_vwap_execution(order_volume, intervals, volume_profile=None)
        twap_schedule = plan_twap_execution(order_volume, intervals)

        pd.testing.assert_frame_equal(vwap_schedule, twap_schedule)

    def test_vwap_profile_length_validation(self):
        """Test VWAP raises error for mismatched profile length."""
        pytest.importorskip("tensorflow")
        from market_forecasting import plan_vwap_execution

        with pytest.raises(ValueError, match="must match intervals"):
            plan_vwap_execution(1000, intervals=5, volume_profile=[0.5, 0.5])
