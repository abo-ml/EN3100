"""Order flow imbalance scalping strategies with broker API integration.

This module provides:
- Order book snapshot fetching via Alpaca or Binance APIs
- Order flow imbalance (OFI) calculations
- Scalping signals based on short-term OFI and mid-price momentum
- Order placement via Alpaca paper trading API

Configuration:
    Set the following environment variables:
    - For Alpaca: APCA_API_KEY_ID, APCA_API_SECRET_KEY
    - For Binance: BINANCE_API_KEY, BINANCE_API_SECRET
    - APCA_API_BASE_URL: (optional) defaults to paper trading URL

Environment variable validation uses check_env_var from src.utils for
consistent error handling and helpful warning messages.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils import check_env_var

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Represents a single order book snapshot.

    Attributes
    ----------
    symbol : str
        The trading symbol (e.g., 'AAPL', 'BTCUSDT').
    timestamp : pd.Timestamp
        Time of the snapshot.
    bids : List[Tuple[float, float]]
        List of (price, size) tuples for bid levels, sorted by price descending.
    asks : List[Tuple[float, float]]
        List of (price, size) tuples for ask levels, sorted by price ascending.
    """
    symbol: str
    timestamp: pd.Timestamp
    bids: List[Tuple[float, float]] = field(default_factory=list)
    asks: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def mid_price(self) -> float:
        """Calculate mid-price from best bid and ask."""
        if not self.bids or not self.asks:
            return 0.0
        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        return (best_bid + best_ask) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if not self.bids or not self.asks:
            return 0.0
        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        return best_ask - best_bid

    @property
    def bid_volume(self) -> float:
        """Total volume on bid side."""
        return sum(size for _, size in self.bids)

    @property
    def ask_volume(self) -> float:
        """Total volume on ask side."""
        return sum(size for _, size in self.asks)

    @property
    def depth_ratio(self) -> float:
        """Ratio of bid volume to total volume (0-1 range)."""
        total = self.bid_volume + self.ask_volume
        if total == 0:
            return 0.5
        return self.bid_volume / total


def fetch_orderbook_snapshot(
    symbol: str,
    top_n: int = 10,
    provider: str = "alpaca",
) -> OrderBookSnapshot:
    """Fetch order book snapshot from broker API.

    Parameters
    ----------
    symbol : str
        Trading symbol. For Alpaca use stock symbols (e.g., 'AAPL').
        For Binance use crypto pairs (e.g., 'BTCUSDT').
    top_n : int, default=10
        Number of price levels to fetch for each side.
    provider : str, default='alpaca'
        API provider to use: 'alpaca' for stocks, 'binance' for crypto.

    Returns
    -------
    OrderBookSnapshot
        Snapshot containing top bid/ask prices and sizes.

    Raises
    ------
    ValueError
        If API keys are not configured or provider is invalid.
    RuntimeError
        If API request fails.

    Environment Variables
    --------------------
    For Alpaca:
        - APCA_API_KEY_ID: Alpaca API key ID
        - APCA_API_SECRET_KEY: Alpaca API secret key
        - APCA_API_BASE_URL: API base URL (optional, defaults to paper trading)

    For Binance:
        - BINANCE_API_KEY: Binance API key (optional for public endpoints)
        - BINANCE_API_SECRET: Binance API secret (optional for public endpoints)
    """
    if provider.lower() == "alpaca":
        return _fetch_alpaca_orderbook(symbol, top_n)
    elif provider.lower() == "binance":
        return _fetch_binance_orderbook(symbol, top_n)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'alpaca' or 'binance'.")


def _fetch_alpaca_orderbook(symbol: str, top_n: int) -> OrderBookSnapshot:
    """Fetch order book from Alpaca Markets API.

    Note: Alpaca's free tier provides limited order book data.
    For full Level 2 data, a paid subscription may be required.

    Returns empty OrderBookSnapshot with warning if APCA_API_KEY_ID or
    APCA_API_SECRET_KEY are not set.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests library required for API calls. Install with: pip install requests")

    # Check for required Alpaca API keys using check_env_var utility
    api_key = check_env_var("APCA_API_KEY_ID", warn_if_missing=False)
    api_secret = check_env_var("APCA_API_SECRET_KEY", warn_if_missing=False)
    base_url = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not api_secret:
        # Return empty DataFrame with warning if no API keys (as per requirements)
        logger.warning(
            "Alpaca API keys not configured. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
            "environment variables for live order book data. Returning empty snapshot."
        )
        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=pd.Timestamp.now(tz="UTC"),
            bids=[],
            asks=[],
        )

    # Alpaca Data API endpoint for quotes/trades
    # Note: Full order book requires Alpaca Pro subscription
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    # Try to get latest quote as a proxy for order book (best bid/ask)
    data_url = "https://data.alpaca.markets/v2"
    quote_endpoint = f"{data_url}/stocks/{symbol}/quotes/latest"

    try:
        response = requests.get(quote_endpoint, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        quote = data.get("quote", {})
        bids = [(quote.get("bp", 0), quote.get("bs", 0))] if quote.get("bp") else []
        asks = [(quote.get("ap", 0), quote.get("as", 0))] if quote.get("ap") else []

        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=pd.Timestamp(quote.get("t", pd.Timestamp.now(tz="UTC"))),
            bids=bids,
            asks=asks,
        )
    except requests.RequestException as e:
        logger.error(f"Alpaca API request failed: {e}")
        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=pd.Timestamp.now(tz="UTC"),
            bids=[],
            asks=[],
        )


def _fetch_binance_orderbook(symbol: str, top_n: int) -> OrderBookSnapshot:
    """Fetch order book from Binance public API.

    Binance provides full order book data without authentication.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests library required for API calls. Install with: pip install requests")

    # Binance public API endpoint - no auth required
    base_url = "https://api.binance.com/api/v3"
    endpoint = f"{base_url}/depth"

    params = {
        "symbol": symbol.upper(),
        "limit": min(top_n, 100),  # Binance max is 5000, but we limit to top_n
    }

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        bids = [(float(price), float(qty)) for price, qty in data.get("bids", [])]
        asks = [(float(price), float(qty)) for price, qty in data.get("asks", [])]

        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=pd.Timestamp.now(tz="UTC"),
            bids=bids[:top_n],
            asks=asks[:top_n],
        )
    except requests.RequestException as e:
        logger.error(f"Binance API request failed: {e}")
        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=pd.Timestamp.now(tz="UTC"),
            bids=[],
            asks=[],
        )


@dataclass
class OrderFlowAlphaModel:
    """Microstructure alpha model for scalping strategies.

    This model uses order flow imbalance (OFI) and mid-price momentum to
    generate trading signals. It stores intraday volume imbalance statistics
    during fitting and produces signals between -1 (strong sell) and 1 (strong buy).

    Attributes
    ----------
    ofi_mean : float
        Mean OFI value from training data.
    ofi_std : float
        Standard deviation of OFI from training data.
    momentum_mean : float
        Mean mid-price momentum from training data.
    momentum_std : float
        Standard deviation of momentum from training data.
    ofi_weight : float
        Weight for OFI component in signal calculation (default 0.6).
    momentum_weight : float
        Weight for momentum component in signal calculation (default 0.4).

    Parameters
    ----------
    ofi_weight : float, default=0.6
        Weight for OFI in signal calculation. Higher values emphasize order flow.
    momentum_weight : float, default=0.4
        Weight for momentum in signal calculation. Higher values emphasize trend.
    """
    ofi_weight: float = 0.6
    momentum_weight: float = 0.4
    ofi_mean: float = field(default=0.0, init=False)
    ofi_std: float = field(default=1.0, init=False)
    momentum_mean: float = field(default=0.0, init=False)
    momentum_std: float = field(default=1.0, init=False)
    _fitted: bool = field(default=False, init=False)

    def fit(self, orderbook_features: pd.DataFrame) -> "OrderFlowAlphaModel":
        """Fit the alpha model using historical order book data.

        Computes and stores statistics for OFI and mid-price momentum that
        are used to normalize signals during prediction.

        Parameters
        ----------
        orderbook_features : pd.DataFrame
            DataFrame with columns:
            - 'ofi': Order flow imbalance values
            - 'mid_price': Mid-price values (for momentum calculation)
            - 'spread': Bid-ask spread (optional)
            - 'depth_ratio': Bid/total volume ratio (optional)

        Returns
        -------
        OrderFlowAlphaModel
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If required columns are missing from input DataFrame.
        """
        required_cols = ["ofi", "mid_price"]
        missing = [col for col in required_cols if col not in orderbook_features.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Store OFI statistics
        ofi = orderbook_features["ofi"].dropna()
        if len(ofi) > 0:
            self.ofi_mean = float(ofi.mean())
            self.ofi_std = float(ofi.std()) if ofi.std() > 0 else 1.0
        else:
            self.ofi_mean = 0.0
            self.ofi_std = 1.0

        # Calculate and store momentum statistics
        mid_price = orderbook_features["mid_price"].dropna()
        if len(mid_price) > 1:
            momentum = mid_price.pct_change().dropna()
            self.momentum_mean = float(momentum.mean())
            self.momentum_std = float(momentum.std()) if momentum.std() > 0 else 1.0
        else:
            self.momentum_mean = 0.0
            self.momentum_std = 1.0

        self._fitted = True
        logger.info(
            f"OrderFlowAlphaModel fitted: OFI mean={self.ofi_mean:.4f}, "
            f"std={self.ofi_std:.4f}; Momentum mean={self.momentum_mean:.6f}, "
            f"std={self.momentum_std:.6f}"
        )

        return self

    def predict_signal(
        self,
        latest_ofi: float,
        spread: float,
        depth_ratio: float,
        momentum: Optional[float] = None,
    ) -> float:
        """Generate a trading signal based on current order flow features.

        Parameters
        ----------
        latest_ofi : float
            Current order flow imbalance value.
            Positive indicates more buying pressure, negative indicates selling.
        spread : float
            Current bid-ask spread. Higher spread reduces signal magnitude.
        depth_ratio : float
            Ratio of bid volume to total volume (0-1 range).
            Values > 0.5 indicate more buying interest.
        momentum : float, optional
            Current mid-price momentum (price change rate).
            If not provided, only OFI is used for signal generation.

        Returns
        -------
        float
            Signal between -1 (strong sell) and 1 (strong buy).
            Values near 0 indicate no clear directional bias.

        Notes
        -----
        The signal is calculated as a weighted combination of:
        1. Normalized OFI (z-score)
        2. Normalized momentum (z-score)
        3. Depth ratio adjustment

        The result is clipped to [-1, 1] and may be dampened by high spreads.
        """
        # Use default statistics if not fitted
        ofi_mean = self.ofi_mean if self._fitted else 0.0
        ofi_std = self.ofi_std if self._fitted else 1.0
        mom_mean = self.momentum_mean if self._fitted else 0.0
        mom_std = self.momentum_std if self._fitted else 1.0

        # Normalize OFI to z-score
        ofi_zscore = (latest_ofi - ofi_mean) / ofi_std

        # Calculate OFI signal component
        ofi_signal = np.tanh(ofi_zscore)  # Squash to [-1, 1]

        # Calculate momentum signal component if provided
        if momentum is not None:
            mom_zscore = (momentum - mom_mean) / mom_std
            mom_signal = np.tanh(mom_zscore)
            ofi_weight = self.ofi_weight
            mom_weight = self.momentum_weight
        else:
            mom_signal = 0.0
            # Shift all weight to OFI if no momentum
            ofi_weight = 1.0
            mom_weight = 0.0

        # Combine signals
        combined_signal = ofi_weight * ofi_signal + mom_weight * mom_signal

        # Adjust for depth imbalance
        # depth_ratio > 0.5 means more bids, which is bullish
        depth_adjustment = (depth_ratio - 0.5) * 0.2  # Max ±0.1 adjustment
        combined_signal += depth_adjustment

        # Dampen signal for wide spreads (less reliable in illiquid markets)
        # Baseline spread threshold: 0.01 (1% of mid-price) is considered wide
        # Spreads wider than this baseline reduce signal confidence
        if spread > 0:
            spread_baseline = 0.01  # 1% spread threshold
            spread_penalty = min(spread / spread_baseline, 1.0) * 0.1
            combined_signal *= (1 - spread_penalty)

        # Clip to [-1, 1]
        return float(np.clip(combined_signal, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Alpaca Order Placement
# ---------------------------------------------------------------------------


def place_alpaca_order(
    symbol: str,
    qty: float,
    side: str,
    order_type: str = "market",
    time_in_force: str = "day",
) -> Dict[str, Any]:
    """Send an order to Alpaca paper trading.

    This function places an order using the Alpaca Trading API. By default, it
    uses the paper trading base URL unless overridden via the APCA_API_BASE_URL
    environment variable.

    Parameters
    ----------
    symbol : str
        The trading symbol (e.g., 'AAPL', 'MSFT').
    qty : float
        Number of shares to trade. Fractional shares are supported.
    side : str
        Order side: 'buy' or 'sell'.
    order_type : str, default='market'
        Order type: 'market', 'limit', 'stop', or 'stop_limit'.
    time_in_force : str, default='day'
        Time in force: 'day', 'gtc', 'opg', 'cls', 'ioc', or 'fok'.

    Returns
    -------
    Dict[str, Any]
        The JSON response from Alpaca containing order details.

    Raises
    ------
    ValueError
        If API keys are not configured or required parameters are invalid.
    RuntimeError
        If the API request fails.

    Environment Variables
    --------------------
    - APCA_API_KEY_ID: Alpaca API key ID (required)
    - APCA_API_SECRET_KEY: Alpaca API secret key (required)
    - APCA_API_BASE_URL: API base URL (optional, defaults to paper trading URL)

    Examples
    --------
    >>> import os
    >>> os.environ["APCA_API_KEY_ID"] = "your_key"
    >>> os.environ["APCA_API_SECRET_KEY"] = "your_secret"
    >>> result = place_alpaca_order("AAPL", 10, "buy")
    >>> print(result["id"])  # Order ID
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests library required for API calls. Install with: pip install requests"
        )

    # Validate side
    if side not in ("buy", "sell"):
        raise ValueError(f"Invalid side '{side}'. Must be 'buy' or 'sell'.")

    # Validate order_type
    valid_order_types = ("market", "limit", "stop", "stop_limit")
    if order_type not in valid_order_types:
        raise ValueError(
            f"Invalid order_type '{order_type}'. Must be one of {valid_order_types}."
        )

    # Validate time_in_force
    valid_tif = ("day", "gtc", "opg", "cls", "ioc", "fok")
    if time_in_force not in valid_tif:
        raise ValueError(
            f"Invalid time_in_force '{time_in_force}'. Must be one of {valid_tif}."
        )

    base_url = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    # Check for required Alpaca API keys using check_env_var utility
    key_id = check_env_var("APCA_API_KEY_ID", warn_if_missing=False)
    secret_key = check_env_var("APCA_API_SECRET_KEY", warn_if_missing=False)

    if not key_id or not secret_key:
        raise ValueError(
            "Alpaca API keys not configured. Set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY environment variables."
        )

    endpoint = f"{base_url}/v2/orders"
    order = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
    }
    headers = {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(endpoint, json=order, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"Alpaca order placement failed: {e}")
        raise RuntimeError(f"Failed to place order: {e}") from e
