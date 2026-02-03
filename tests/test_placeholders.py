import numpy as np
import pandas as pd
import pytest


def test_orderflow_model_fits_and_predicts():
    """Test that OrderFlowAlphaModel can be fitted and predict signals."""
    from src.advanced.orderflow_scalping import OrderFlowAlphaModel

    # Create sample orderbook features
    data = pd.DataFrame({
        "ofi": np.random.randn(100),
        "mid_price": 100 + np.random.randn(100).cumsum() * 0.1,
        "spread": np.abs(np.random.randn(100)) * 0.01,
        "depth_ratio": np.random.uniform(0.3, 0.7, 100),
    })

    model = OrderFlowAlphaModel()

    # Should fit without error
    model.fit(data)
    assert model._fitted is True

    # Should predict a signal between -1 and 1
    signal = model.predict_signal(0.1, 0.01, 0.5, momentum=0.001)
    assert -1.0 <= signal <= 1.0

    # Should work without momentum
    signal_no_mom = model.predict_signal(0.1, 0.01, 0.5)
    assert -1.0 <= signal_no_mom <= 1.0


def test_pattern_functions_work():
    """Test that pattern detection functions now work instead of raising."""
    from src.advanced.pattern_recognition import detect_double_top, detect_head_and_shoulders

    df = pd.DataFrame({"close": [100.0] * 50})

    # Should now return Series instead of raising NotImplementedError
    hs_result = detect_head_and_shoulders(df)
    assert isinstance(hs_result, pd.Series)
    assert len(hs_result) == 50

    dt_result = detect_double_top(df)
    assert isinstance(dt_result, pd.Series)
    assert len(dt_result) == 50


def test_liquidity_grab_detection():
    """Test liquidity grab detection function."""
    from src.advanced.pattern_recognition import flag_liquidity_grab

    # Create sample price data with a potential liquidity grab
    df = pd.DataFrame({
        "high": [100, 101, 102, 110, 103, 101],  # Spike at index 3
        "low": [98, 99, 100, 95, 100, 99],  # Wide range at index 3
        "close": [99, 100, 101, 100, 102, 100],  # Reversal at index 3
        "volume": [1000, 1000, 1000, 5000, 1000, 1000],  # Volume spike at index 3
    })

    result = flag_liquidity_grab(df)
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.name == "liquidity_grab"
    # Values should be 0 or 1
    assert set(result.unique()).issubset({0, 1})


def test_fvg_detection():
    """Test fair value gap detection function."""
    from src.advanced.pattern_recognition import detect_fvg

    # Create sample price data with a bullish FVG
    # Candle 0: high=100
    # Candle 1: normal
    # Candle 2: low=102 (gap above candle 0's high)
    df = pd.DataFrame({
        "high": [100, 101, 105, 106, 107],
        "low": [98, 99, 102, 103, 104],
    })

    result = detect_fvg(df)
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.name == "fvg"
    # Values should be -1, 0, or 1
    assert set(result.unique()).issubset({-1, 0, 1})


def test_asia_session_breakout():
    """Test Asia session breakout detection function."""
    from src.advanced.pattern_recognition import asia_session_range_breakout

    # Create sample daily data
    df = pd.DataFrame({
        "high": [100, 102, 105, 103, 101],
        "low": [98, 99, 100, 101, 99],
        "close": [99, 101, 104, 102, 100],
    }, index=pd.date_range("2023-01-01", periods=5, freq="D"))

    result = asia_session_range_breakout(df)
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.name == "asia_breakout"
    # Values should be -1, 0, or 1
    assert set(result.unique()).issubset({-1, 0, 1})


def test_rl_env_creates_and_runs():
    """Test TradingEnv can be created and stepped through."""
    pytest.importorskip("gymnasium")
    from src.advanced.reinforcement_learning import TradingEnv, TradingEnvConfig

    # Create sample data
    prices = np.random.randn(200).cumsum() + 100
    features = np.random.randn(200, 3)

    config = TradingEnvConfig(window_size=30, transaction_cost=0.0001)
    env = TradingEnv(prices=prices, features=features, config=config)

    # Test reset
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert "equity" in info

    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "equity" in info
    assert "drawdown" in info
