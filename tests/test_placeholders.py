import pandas as pd
import pytest


def test_orderflow_placeholders_raise():
    from src.advanced.orderflow_scalping import OrderFlowAlphaModel

    m = OrderFlowAlphaModel()
    with pytest.raises(NotImplementedError):
        m.fit(pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(NotImplementedError):
        m.predict_signal(0.1, 0.01, 0.5)


def test_pattern_placeholders_raise():
    from src.advanced.pattern_recognition import detect_double_top, detect_head_and_shoulders

    with pytest.raises(NotImplementedError):
        detect_head_and_shoulders(pd.DataFrame({"Close": [1, 2, 3]}))
    with pytest.raises(NotImplementedError):
        detect_double_top(pd.DataFrame({"Close": [1, 2, 3]}))


def test_rl_env_placeholder_guard():
    gym = pytest.importorskip("gym")
    from src.advanced.reinforcement_learning import TradingEnv

    with pytest.raises(NotImplementedError):
        TradingEnv()
