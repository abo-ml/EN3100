"""Future-work module for order flow imbalance scalping strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class OrderFlowAlphaModel:
    """Placeholder microstructure alpha model.

    Intended for high-frequency or sub-hour strategies that exploit order flow
    imbalance and liquidity dynamics. In future work this class should ingest
    Level 2 order book features, compute order flow imbalance (OFI), and produce
    scalp signals for rapid execution strategies.
    """

    def fit(self, orderbook_features: Any) -> None:  # noqa: D401
        """TODO: Fit the alpha model using historical order book data."""
        raise NotImplementedError(
            "OrderFlowAlphaModel.fit() is not implemented. "
            "Requires Level 2 order book data from a broker API."
        )

    def predict_signal(self, latest_ofi: float, spread: float, depth_ratio: float) -> int:
        """Return a placeholder scalp signal based on OFI inputs."""
        raise NotImplementedError(
            "OrderFlowAlphaModel.predict_signal() is not implemented. "
            "Requires real-time order flow imbalance features."
        )
