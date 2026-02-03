import pandas as pd
import pytest


def test_orderflow_placeholders_raise():
    from src.advanced.orderflow_scalping import OrderFlowAlphaModel

    m = OrderFlowAlphaModel()
    with pytest.raises(NotImplementedError):
        m.fit(pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(NotImplementedError):
        m.predict_signal(0.1, 0.01, 0.5)


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


def test_rl_env_placeholder_guard():
    gym = pytest.importorskip("gym")
    from src.advanced.reinforcement_learning import TradingEnv

    with pytest.raises(NotImplementedError):
        TradingEnv()
